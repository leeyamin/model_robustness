import os
import re
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import argparse

class MedicalQASummarizer:
    """A class to process and summarize medical Q&A pairs into concise question-answer format."""
    
    def __init__(self, max_words: int = 40, min_sentence_length: int = 5):
        """
        Initialize the summarizer.
        
        Args:
            max_words: Maximum number of words for the summary
            min_sentence_length: Minimum length of a sentence to be included
        """
        self.max_words = max_words
        self.min_sentence_length = min_sentence_length
        self.greetings = {'hi', 'hello', 'dear', 'thanks', 'thank you', 'regards', 'sincerely', 'best', 'warm'}
        self.key_terms = {
            # Action terms
            'should', 'recommend', 'advise', 'suggest', 'take', 'use', 'avoid', 'consider', 'try',
            'important', 'necessary', 'required', 'need', 'must', 'essential', 'typically', 'usually',
            'often', 'commonly', 'frequently', 'prescribe', 'prescription', 'treatment', 'therapy',
            # Medical terms
            'cause', 'causes', 'caused', 'due to', 'because of', 'result from', 'lead to', 'leads to',
            'symptom', 'symptoms', 'sign', 'signs', 'indication', 'indications', 'manifest', 'manifestation',
            'diagnose', 'diagnosis', 'diagnosed', 'identify', 'identification', 'test', 'testing',
            'treat', 'treatment', 'therapy', 'therapies', 'medication', 'medications', 'medicine', 'medicines',
            'drug', 'drugs', 'prescription', 'prescriptions', 'dose', 'dosage', 'dosages', 'mg', 'ml',
            'prevent', 'prevention', 'preventive', 'avoid', 'avoiding', 'reduction', 'reduce', 'reducing',
            'risk', 'risks', 'factor', 'factors', 'complication', 'complications', 'side effect', 'side effects',
            # Time-related terms
            'acute', 'chronic', 'sudden', 'gradual', 'immediate', 'delayed', 'prolonged', 'temporary', 'permanent',
            # Severity terms
            'mild', 'moderate', 'severe', 'critical', 'serious', 'minor', 'major', 'life-threatening', 'fatal',
            # Location terms
            'localized', 'generalized', 'systemic', 'bilateral', 'unilateral', 'left', 'right', 'upper', 'lower'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the text, removing personal references and greetings."""
        if not isinstance(text, str):
            return ""
        
        text = ' '.join(text.split())  # Remove extra whitespace
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        text = re.split(r'(?i)(?:sincerely|best regards|regards|thank you|thanks)[,.]', text)[0]
        
        text = re.sub(r'\b(I|me|my|mine|myself|you|your|yours|yourself|yourselves)\b', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'^\s*(hi|hello|dear|hey)[,.!\s]*', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_greeting_or_signoff(self, sentence: str) -> bool:
        """Check if a sentence is a greeting, sign-off, or contains personal references."""
        if not sentence.strip():
            return True
            
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        if len(words) < 3:
            return True
            
        greeting_phrases = [
            'best regards', 'sincerely', 'thank you', 'thanks', 'hope this helps',
            'take care', 'let me know', 'feel free', 'for more information',
            'consult a', 'see a', 'visit a', 'contact your', 'dear', 'hi ', 'hello ',
            'warm regards', 'kind regards', 'yours sincerely', 'yours truly'
        ]
        
        if any(phrase in sentence_lower for phrase in greeting_phrases):
            return True
            
        personal_pronouns = ['i ', 'me ', 'my ', 'mine', 'myself', 'you ', 'your', 'yours', 'yourself']
        if any(pronoun in sentence_lower for pronoun in personal_pronouns):
            return True
            
        if '?' in sentence:
            return True
            
        return False
    
    def score_sentence(self, sentence: str) -> int:
        """Score a sentence based on its content and relevance."""
        if not sentence.strip():
            return -1
            
        score = 0
        words = sentence.lower().split()
        sentence_lower = sentence.lower()
        
        score += sum(1 for term in self.key_terms if term in sentence_lower)
        
        word_count = len(words)
        if 8 <= word_count <= 25:
            score += 2
        elif 4 <= word_count <= 7 or 26 <= word_count <= 35:
            score += 1
            
        if word_count < 3 or word_count > 50:
            score -= 2
            
        if sentence.count(',') > 3 or sentence.count(';') > 1:
            score -= 1
            
        if 'http' in sentence_lower or '@' in sentence_lower:
            score -= 2
            
        return score
    
    def generate_summary(self, text: str) -> str:
        """Generate a meaningful summary from the given text."""
        if not text:
            return ""
            
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        filtered_sentences = [
            s for s in sentences 
            if not self.is_greeting_or_signoff(s) and 
               len(s.split()) >= self.min_sentence_length
        ]
        
        if not filtered_sentences:
            filtered_sentences = sentences
        
        scored_sentences = [(s, self.score_sentence(s)) for s in filtered_sentences]
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        summary = []
        word_count = 0
        
        for sentence, score in scored_sentences:
            words = sentence.split()
            if word_count + len(words) <= self.max_words:
                summary.append(sentence)
                word_count += len(words)
            else:
                remaining = self.max_words - word_count
                if remaining >= 3:
                    partial = ' '.join(words[:remaining]) + '...'
                    summary.append(partial)
                break
        
        if not summary and sentences:
            first_sentence = sentences[0]
            words = first_sentence.split()
            return ' '.join(words[:min(len(words), self.max_words)]) + ('...' if len(words) > self.max_words else '')
        
        return ' '.join(summary)
    
    def extract_key_information(self, description: str, doctor_response: str) -> str:
        """
        Extract key information from the doctor's response that is relevant to the description.
        Returns a neutral, concise answer based on the doctor's literal response.
        
        Args:
            description: The concise description of the question
            doctor_response: The doctor's full response
            
        Returns:
            str: A concise, neutral answer based on the doctor's response
        """
        if not doctor_response or not description:
            return ""
            
        clean_description = self.clean_text(description)
        
        doc_response_cleaned = ' '.join(doctor_response.split())
        doc_response_cleaned = re.sub(r'https?://\S+|www\.\S+', '', doc_response_cleaned)
        doc_response_cleaned = re.sub(r'<.*?>', '', doc_response_cleaned)
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc_response_cleaned)]
        
        prepped_sentences = []
        for sentence in sentences:
            if self.is_greeting_or_signoff(sentence) or len(sentence.split()) < self.min_sentence_length:
                continue
            
            cleaned_sentence = re.sub(r'\b(I|me|my|mine|myself|you|your|yours|yourself|yourselves)\b', '', sentence, flags=re.IGNORECASE)
            cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
            
            if cleaned_sentence:
                prepped_sentences.append(cleaned_sentence)

        if not prepped_sentences:
            return ""
            
        question_type = self._identify_question_type(clean_description)
        
        scored_sentences = [
            (s, self._calculate_relevance_score(s, clean_description, question_type)) 
            for s in prepped_sentences
        ]
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        answer_parts = []
        word_count = 0
        
        for sentence, score in scored_sentences:
            if score <= 0:
                continue

            words = sentence.split()
            if word_count + len(words) <= self.max_words:
                answer_parts.append(sentence)
                word_count += len(words)
            elif not answer_parts:
                answer_parts.append(' '.join(words[:self.max_words]) + '...')
                break
        
        if not answer_parts and prepped_sentences:
            first_sentence = prepped_sentences[0]
            words = first_sentence.split()
            result = ' '.join(words[:min(len(words), self.max_words)])
            if len(words) > self.max_words:
                result += '...'
            return self._clean_answer(result)
            
        answer = ' '.join(answer_parts)
        return self._clean_answer(answer)

    def _identify_question_type(self, description: str) -> str:
        """Identify the type of question based on the description."""
        description = description.lower()
        if any(word in description for word in ['what cause', 'what causes', 'why do', 'why does']):
            return 'cause'
        elif any(word in description for word in ['what are the symptom', 'what are the sign']):
            return 'symptoms'
        elif any(word in description for word in ['how to treat', 'what is the treatment', 'how is it treated']):
            return 'treatment'
        elif any(word in description for word in ['how to prevent', 'how can i prevent']):
            return 'prevention'
        elif any(word in description for word in ['what is', 'what are', 'define']):
            return 'definition'
        return 'general'

    def _calculate_relevance_score(self, sentence: str, description: str, question_type: str) -> float:
        """Calculate how relevant a sentence is to the question and its type."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        score += sum(0.5 for term in self.key_terms if term in sentence_lower)
        
        description_terms = set(description.lower().split())
        sentence_terms = set(sentence_lower.split())
        common_terms = description_terms.intersection(sentence_terms)
        score += len(common_terms) * 0.5  # Boost for each matching term
        
        if question_type == 'cause' and ('cause' in sentence_lower or 'due to' in sentence_lower):
            score += 2
        elif question_type == 'symptoms' and ('symptom' in sentence_lower or 'sign' in sentence_lower):
            score += 2
        elif question_type == 'treatment' and ('treat' in sentence_lower or 'therapy' in sentence_lower or 'medication' in sentence_lower):
            score += 2
        elif question_type == 'prevention' and ('prevent' in sentence_lower or 'avoid' in sentence_lower):
            score += 2
            
        word_count = len(sentence_lower.split())
        if word_count < 3 or word_count > 30:
            score -= 1
            
        return max(0, score)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up the final answer."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)]
        cleaned_sentences = [s for s in sentences if not self.is_greeting_or_signoff(s)]
        
        if not cleaned_sentences:
            return answer
            
        result = ' '.join(cleaned_sentences)
        result = re.sub(r'\s+', ' ', result).strip()
        
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
            
        return result
    
    def generate_answer(self, description: str, doctor_response: str) -> str:
        """
        Generate a concise answer from the doctor's response based on the description.
        
        Args:
            description: The concise description of the question
            doctor_response: The doctor's full response
            
        Returns:
            str: A concise answer
        """
        try:
            answer = self.extract_key_information(description, doctor_response)
            
            if not answer:
                return "[No answer could be generated]"
                
            return answer.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "[Error generating answer]"

def process_batch(
    dataset,
    processor: MedicalQASummarizer,
    start_idx: int,
    batch_size: int,
    output_dir: str,
    batch_num: int
) -> Tuple[pd.DataFrame, str]:
    """
    Process a batch of the dataset.
    
    Args:
        dataset: The dataset to process
        processor: The MedicalQASummarizer instance
        start_idx: Starting index in the dataset
        batch_size: Number of samples to process
        output_dir: Directory to save results
        batch_num: Batch number for logging
        
    Returns:
        Tuple containing the processed DataFrame and output file path
    """
    end_idx = min(start_idx + batch_size, len(dataset))
    batch_results = []
    
    image_keywords = re.compile(r'image|photo|picture|attachment|attached|view the', re.IGNORECASE)
    
    for idx in range(start_idx, end_idx):
        try:
            item = dataset[idx]
            
            description = item.get('Description', '')
            patient_query = item.get('Patient', '')
            doctor_response = item.get('Doctor', '')
            
            if image_keywords.search(patient_query) or image_keywords.search(doctor_response):
                continue
            
            answer = processor.generate_answer(
                description=description,
                doctor_response=doctor_response
            )
            
            question = re.sub(r'^Q\.\s*', '', description).strip()
            
            result = {
                'question': question,
                'patient_query': patient_query,
                'doctor_response': doctor_response,
                'concise_answer': answer,
            }
            
            batch_results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    df = pd.DataFrame(batch_results)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'medical_qa_batch_{batch_num:03d}.csv')
    
    df.to_csv(output_file, index=False)
    
    return df, output_file

def main():
    parser = argparse.ArgumentParser(description='Process medical Q&A dataset and generate concise answers.')
    parser.add_argument('--output-dir', type=str, default='data/processed/medical_qa',
                        help='Directory to save processed data')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of samples to process in each batch')
    parser.add_argument('--max-words', type=int, default=40,
                        help='Maximum number of words in the summary')
    
    args = parser.parse_args()
    
    try:
        print("Initializing MedicalQASummarizer...")
        processor = MedicalQASummarizer(max_words=args.max_words)
        
        print("Loading dataset...")
        dataset = load_dataset('ruslanmv/ai-medical-chatbot', split='train')
        original_size = len(dataset)
        print(f"Loaded {original_size} samples")
        
        start_index = 0
        end_idx = original_size
        total_samples = end_idx - start_index
        
        print(f"Processing {total_samples} samples...")
        
        all_results = []
        
        with tqdm(total=total_samples, desc="Processing dataset") as pbar:
            batch_num = 1
            for start in range(start_index, end_idx, args.batch_size):
                batch_end = min(start + args.batch_size, end_idx)
                
                df_batch, _ = process_batch(
                    dataset=dataset,
                    processor=processor,
                    start_idx=start,
                    batch_size=args.batch_size,
                    output_dir=args.output_dir,
                    batch_num=batch_num
                )
                
                if not df_batch.empty:
                    all_results.append(df_batch)
                
                pbar.update(batch_end - start)
                batch_num += 1
        
        if all_results:
            df_final = pd.concat(all_results, ignore_index=True)
            
            answered_mask = (df_final['concise_answer'] != '[No answer could be generated]') & \
                            (~df_final['concise_answer'].str.contains('revert back', case=False, na=False))
            df_final_answered = df_final[answered_mask]
            
            final_output_file = os.path.join(args.output_dir, 'medical_qa_final.csv')
            df_final_answered.to_csv(final_output_file, index=False)
            
            for batch_file in os.listdir(args.output_dir):
                if batch_file.startswith('medical_qa_batch_') and batch_file.endswith('.csv'):
                    try:
                        os.remove(os.path.join(args.output_dir, batch_file))
                    except Exception as e:
                        print(f"Could not delete batch file {batch_file}: {e}")
            
            print("\nProcessing complete!")
            print(f"Original dataset size: {original_size}")
            print(f"Records with generated answers: {len(df_final_answered)}")
            print(f"Final results saved to: {final_output_file}")
        
        else:
            print("Warning: No results were generated.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
