import os
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import List
import argparse
import torch
from transformers import pipeline

def remove_greetings_and_closings(text: str, greeting_terms: List[str] = None) -> str:
    """
    Remove greeting and closing terms from text without modifying the medical content.
    Only removes exact matches of specified terms (case-insensitive).
    """
    if greeting_terms is None:
        greeting_terms = [
            'hi', 'hello', 'hey', 'dear', 'thanks', 'thank you',
            'regards', 'sincerely', 'best wishes', 'goodbye', 'bye',
            'welcome', 'greetings', '-->', 'good morning', 'good afternoon',
            'good evening', 'have a good day'
        ]
    
    text_lower = text.lower()
    words = text.split()
    words_lower = text_lower.split()
    
    filtered_words = []
    for i, (word, word_lower) in enumerate(zip(words, words_lower)):
        is_greeting = False
        for term in greeting_terms:
            term_len = len(term.split())
            if i <= len(words) - term_len:
                phrase = ' '.join(words_lower[i:i+term_len])
                if phrase == term:
                    is_greeting = True
                    break
        
        if not is_greeting:
            filtered_words.append(word)
    
    return ' '.join(filtered_words).strip()

def is_referral_only(answer: str) -> bool:
    """
    Check if the answer is just a referral without any medical information.
    """
    referral_patterns = [
        r"(?i)consult.*online",
        r"(?i)please.*consult",
        r"(?i)visit.*doctor",
        r"(?i)see.*specialist",
    ]
    
    is_referral = any(re.search(pattern, answer) for pattern in referral_patterns)
    
    words = set(answer.lower().split())
    stop_words = {"hi", "hello", "please", "thank", "thanks", "regards", "sincerely", "for", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of"}
    content_words = words - stop_words
    
    return is_referral and len(content_words) < 5

def clean_text(text: str) -> str:
    """
    Clean text by fixing common typos and formatting issues.
    """
    typo_fixes = {
        'ypu': 'you',
        'acud': 'acid',
        'aluke': 'like',
        'He ce': 'Hence',
        'threatns': 'threatens',
        'differed': 'suffered',
        ' i ': ' I ',
        'with in': 'within',
    }
    
    for typo, fix in typo_fixes.items():
        text = re.sub(rf'\b{typo}\b', fix, text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    
    text = re.sub(r'^(?:Doctor\'s Answer:|Answer:|Response:|Medical Advice:)\s*', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation and extra whitespace.
    """
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def verify_no_added_content(original: str, summary: str) -> bool:
    """
    Verify that the summary doesn't contain any medical terms or key phrases that weren't in the original text.
    Returns True if the summary is valid (no new content), False otherwise.
    """
    original_norm = normalize_text(original)
    summary_norm = normalize_text(summary)
    
    def extract_terms(text):
        potential_terms = re.findall(r'\b[a-z]+(?:itis|osis|emia|opathy|ectomy|plasty|therapy|tomy|scopy|gram)\b', text)
        
        words = re.findall(r'\b[a-z]{7,}\b', text)
        potential_terms.extend(words)
        
        measurements = re.findall(r'\d+(?:\.\d+)?\s*(?:mg|g|ml|l|cm|mm|hours?|days?|weeks?|months?)', text)
        
        all_words = text.split()
        
        phrases = []
        for window_size in range(2, 5):
            for i in range(len(all_words) - window_size + 1):
                phrase = ' '.join(all_words[i:i + window_size])
                if (any(len(word) > 6 for word in phrase.split()) or
                    re.search(r'\b(?:treatment|symptom|condition|diagnosis|therapy|medicine|disease|infection|surgery)\b', phrase, re.IGNORECASE)):
                    phrases.append(phrase)
        
        return set(potential_terms + measurements + phrases)
    
    original_terms = extract_terms(original_norm)
    summary_terms = extract_terms(summary_norm)
    
    new_terms = set()
    for term in summary_terms:
        if term in {'there are', 'this is', 'it is', 'you are', 'will be', 'can be', 'has been'}:
            continue
            
        term_norm = normalize_text(term)
        
        if len(term.split()) > 2:
            term_words = set(term.split())
            if all(word in original_norm.split() for word in term_words):
                continue
                
        if term_norm not in original_norm and not any(term_norm in normalize_text(orig_term) for orig_term in original_terms):
            term_words = term_norm.split()
            found = False
            for i in range(len(original_norm.split()) - len(term_words) + 1):
                if ' '.join(original_norm.split()[i:i + len(term_words)]) == ' '.join(term_words):
                    found = True
                    break
            if not found:
                new_terms.add(term)
    
    if new_terms:
        print(f"Found new terms in summary that weren't in original: {new_terms}")
        return False
    
    return True

def generate_concise_answer(summarizer, question: str, full_answer: str) -> str:
    """
    Generate a concise answer from the doctor's full response using a summarization model.
    Ensures the summary contains only information present in the original answer.
    """
    full_answer = clean_text(full_answer)
    
    if len(full_answer.split()) < 10 or is_referral_only(full_answer):
        print(f"\nSkipping answer as it's too short or referral-only:")
        print(f"Question: {question}")
        print(f"Answer: {full_answer}")
        return None
        
    prompt = f"""Summarize this medical response, focusing ONLY on medical information, advice, and recommendations:

Question: {question}
Doctor's Answer: {full_answer}

Guidelines:
- Keep ALL medical terms, conditions, symptoms, and treatments EXACTLY as they appear
- Keep ALL specific recommendations and dosages EXACTLY as they appear
- Remove greetings, pleasantries, and non-medical content
- Make it direct and concise while preserving medical accuracy
- Start with the most important medical information
- Ensure all sentences are complete and end with proper punctuation
- Do not cut off sentences in the middle
- Keep the most relevant medical information for the question asked
- Do not add ANY new information that wasn't in the original text
- Do not add ANY labels or prefixes to the text"""

    try:
        input_length = len(full_answer.split())
        max_length = min(max(input_length * 2 // 3, 40), 175)
        min_length = min(max(input_length // 3, 25), 125)
        
        print(f"\nProcessing QA pair:")
        print(f"Question: {question}")
        print(f"Original answer length: {len(full_answer.split())} words")
        print(f"Target summary length: {min_length}-{max_length} words")
        
        summary = summarizer(
            prompt, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=True,
            repetition_penalty=2.5
        )
        concise_answer = clean_text(summary[0]['summary_text'])
        
        if any(phrase in concise_answer.lower() for phrase in [
            "keep all medical terms",
            "remove greetings",
            "remove any greetings",
            "guidelines",
            "summarize this",
            "medical information",
            "medical response",
            "doctor's answer",
            "original text"
        ]):
            print("Summary contained instruction text, skipping...")
            return None
            
        if not concise_answer[-1] in '.!?':
            sentences = re.split(r'(?<=[.!?])\s+', concise_answer)
            if sentences:
                concise_answer = ' '.join(sentences[:-1] if len(sentences) > 1 else sentences)
                if not concise_answer[-1] in '.!?':
                    concise_answer = concise_answer.rstrip(',:;') + '.'
        
        if len(concise_answer.split()) < min(20, min_length):
            print("Summary too short after cleaning, skipping...")
            return None
            
        if not verify_no_added_content(full_answer, concise_answer):
            print("Summary contained new content not present in original, skipping...")
            return None
        
        print(f"Generated summary length: {len(concise_answer.split())} words")
        print(f"Original answer: {full_answer}")
        print(f"Concise answer: {concise_answer}\n")
        
        return concise_answer
    except Exception as e:
        print(f"Error generating concise answer: {str(e)}")
        return None

def clean_question_text(text: str) -> str:
    """
    Clean question text by removing the Q. prefix and any leading/trailing whitespace.
    """
    text = re.sub(r'^Q\.\s*', '', text.strip())
    return text.strip()

def process_medical_qa(
    dataset, model_name: str = "facebook/bart-large-cnn", output_dir: str = "data/processed/medical_qa",
    batch_size: int = 10) -> pd.DataFrame:

    print("Loading summarization model...")
    summarizer = pipeline("summarization", model=model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")
    
    processed_data = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing QA pairs"):
        batch = dataset[i:i + batch_size]
        
        for item in zip(batch['Description'], batch['Doctor']):
            question = clean_question_text(item[0])
            answer = item[1].strip()
            
            concise_answer = generate_concise_answer(summarizer, question, answer)
            
            if concise_answer is not None:
                processed_data.append({
                    'question': question,
                    'original_answer': answer,
                    'concise_answer': concise_answer
                })
    
    df = pd.DataFrame(processed_data)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'medical_qa_processed.csv')
    df.to_csv(output_file, index=False)
    
    print("\nProcessing complete!")
    print(f"Original dataset size: {len(dataset)}")
    print(f"Processed QA pairs: {len(df)}")
    print(f"Results saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Process medical Q&A dataset.')
    parser.add_argument('--output-dir', type=str, default='data/processed/medical_qa',
                        help='Directory to save processed data')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of samples to process in each batch')
    parser.add_argument('--model-name', type=str, default="facebook/bart-large-cnn",
                        help='Model to use for generating concise answers')
    
    args = parser.parse_args()
    
    try:
        print("Loading dataset...")
        dataset = load_dataset('ruslanmv/ai-medical-chatbot', split='train')

        dataset = dataset.select(range(20000))
        
        df = process_medical_qa(
            dataset=dataset,
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
