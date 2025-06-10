import os
import argparse
import random
import re
import subprocess

import config
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import nltk
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

load_dotenv()
nltk.download('punkt', quiet=True)

def set_seeds(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

set_seeds()


def load_medical_questions(source='medical_qa', num_questions=config.NUM_QUESTIONS):
    """Load questions and answers from the specified source."""
    if source == 'poc':
        print("Loading POC data source...")
        return config.POC_DATA['questions'][:num_questions], config.POC_DATA['answers'][:num_questions]

    if not os.path.exists(config.MEDICAL_QA_PATH):
        print(f"Medical QA CSV not found at {config.MEDICAL_QA_PATH}. Running medical_qa_processor.py...")
        try:
            subprocess.run(["python", "medical_qa_processor.py"], check=True)
            if not os.path.exists(config.MEDICAL_QA_PATH):
                raise FileNotFoundError(f"Failed to generate {config.MEDICAL_QA_PATH}")
        except Exception as e:
            print(f"Error running medical_qa_processor.py: {e}")
            return [], []

    try:
        df = pd.read_csv(config.MEDICAL_QA_PATH, nrows=num_questions)
        questions = df['question'].tolist()
        concise_answers = df['concise_answer'].tolist()
        return questions, concise_answers
    except Exception as e:
        print(f"Error loading medical questions: {e}")
        return [], []


def preprocess_text(text):
    """Basic text preprocessing for better similarity matching."""
    if not text:
        return ""
    text = str(text)
    text = ' '.join(text.split()).lower()
    fillers = [
        "according to the context", "based on the information provided",
        "the text states that", "it is mentioned that", "as described",
        "in summary", "to summarize", "in conclusion"
    ]
    for filler in fillers:
        text = text.replace(filler, '')
    return text.strip()

def get_text_embeddings(texts):
    """Generate sentence embeddings for a list of texts."""
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    return model.encode(texts)

def remove_near_duplicates(texts, threshold=0.92):
    """Remove near-duplicate texts based on semantic similarity."""
    embeddings = get_text_embeddings(texts)
    keep = []
    kept_indices = []
    for i, emb in enumerate(embeddings):
        is_dup = False
        for j in kept_indices:
            sim = util.pytorch_cos_sim(np.array(emb), np.array(embeddings[j])).item()
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(texts[i])
            kept_indices.append(i)
    return keep

def remove_generic_answers(answers):
    """Filter out predefined generic or unhelpful answers."""
    def is_generic(ans):
        a = ans.strip().lower()
        return any(gen.lower() in a for gen in config.GENERIC_ANSWERS)
    return [a for a in answers if not is_generic(a)]

def clean_answers(answers):
    """Apply all cleaning steps to a list of answers."""
    answers = remove_generic_answers(answers)
    answers = remove_near_duplicates(answers, threshold=0.92)
    return answers


def get_generation_config():
    """Return a config dict for model generation."""
    return {
        'max_new_tokens': 512,
        'temperature': 0.7,
        'context_length': config.CONTEXT_LENGTH,
        'stop': ["\nQ:", "\nQuestion:", "Question:"]
    }

def count_tokens(text):
    """Count tokens in a string (basic whitespace split)."""
    return len(text.split())

def truncate_prompt_to_fit(prompt_template_str, user_input, max_context_tokens):
    """Truncates user_input to fit within the model's context window."""
    dummy_input = "PLACEHOLDER"
    try:
        prompt_with_dummy = prompt_template_str.format(Question=dummy_input)
        prompt_tokens = count_tokens(prompt_with_dummy) - count_tokens(dummy_input)
    except (KeyError, ValueError) as e:
        print(f"[WARNING] Malformed prompt template. Estimating token count from raw string: {e}")
        prompt_tokens = count_tokens(prompt_template_str)

    allowed_input_tokens = max_context_tokens - prompt_tokens
    if allowed_input_tokens <= 0:
        print(f"[WARNING] Prompt template alone exceeds context length. No input will be used.")
        return ""

    tokens = user_input.split()
    if len(tokens) > allowed_input_tokens:
        return ' '.join(tokens[:allowed_input_tokens])
    return user_input

def _get_llm_instance(model_name):
    """Initializes and returns a CTransformers LLM instance."""
    config_dict = get_generation_config()
    kwargs = {'config': config_dict, 'threads': os.cpu_count() or 1}
    name_lower = model_name.lower()
    if 'llama' in name_lower:
        kwargs['model_type'] = 'llama'
    elif 'mistral' in name_lower:
        kwargs['model_type'] = 'mistral'
    return CTransformers(model=model_name, **kwargs)

def get_model_answer(prompt, model_name, max_attempts=3):
    """Generate a clean answer from a model, with retries for empty/bad responses."""
    prompt_template_str = "Answer concisely and factually in three sentences or less: {Question}"
    prompt_template = PromptTemplate(input_variables=["Question"], template=prompt_template_str)
    
    llm = _get_llm_instance(model_name)
    chain = prompt_template | llm
    
    truncated_prompt = truncate_prompt_to_fit(prompt_template_str, prompt, config.CONTEXT_LENGTH)
    if not truncated_prompt and prompt:
        return "Prompt was too long for context and could not be processed."

    for _ in range(max_attempts):
        raw_answer = chain.invoke({"Question": truncated_prompt})
        if isinstance(raw_answer, dict) and 'text' in raw_answer:
            raw_answer = raw_answer['text']
        
        answer = raw_answer.strip()

        answer = re.split(r'\\n\\s*(Q|A|Question|Answer):', answer)[0].strip()

        full_prompt_sent = prompt_template.format(Question=truncated_prompt)
        if answer.lower().startswith(full_prompt_sent.lower().strip()):
            answer = answer[len(full_prompt_sent):].strip()
        elif answer.lower().startswith(truncated_prompt.lower().strip()):
             answer = answer[len(truncated_prompt):].strip()

        filler_patterns = [
            r"^(Sure|Of course|Certainly),? here (is|'s) (the|a concise) answer:?",
            r"^Here is a concise and factual answer:?",
            r"^(Here's|Here is) the answer:?",
            r"^(Answer|A)\\s*[:：]?",
            r"^Answer concisely\\s*[:.]?"
        ]
        for pattern in filler_patterns:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).lstrip(' :.\\n').strip()

        if answer:
            return answer
    
    return "No answer available."

def get_robust_model_answer(prompt, model_name, max_retries=3):
    """
    Wrapper for get_model_answer that retries if the answer is "No answer available.".
    """
    for i in range(max_retries):
        answer = get_model_answer(prompt, model_name)
        if answer != "No answer available.":
            return answer
        if i < max_retries - 1:
            print(f"    [INFO] Answer was unavailable. Retrying ({i+1}/{max_retries-1})...")
    return "No answer available."

def diverse_probes(probes, num_desired, min_clusters=2):
    """Select a diverse subset of probes using clustering."""
    if len(probes) <= num_desired:
        return probes
        
    embeddings = get_text_embeddings(probes)
    
    if len(embeddings) <= num_desired:
        return probes
        
    n_clusters = min(num_desired, len(probes), max(min_clusters, 2))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    
    selected_probes = []
    selected_indices = []
    for cluster_id in range(n_clusters):
        indices_in_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if indices_in_cluster:
            selected_index = indices_in_cluster[0]
            selected_indices.append(selected_index)
            selected_probes.append(probes[selected_index])
    
    if len(selected_probes) < num_desired:
        sel_embeddings = np.array([embeddings[i] for i in selected_indices])
        
        remaining_indices = [i for i in range(len(probes)) if i not in selected_indices]
        
        remaining_probes_with_dist = []
        for i in remaining_indices:
            rem_embedding = embeddings[i]
            if sel_embeddings.any():
                sims = util.pytorch_cos_sim(rem_embedding, sel_embeddings)[0]
                min_dist = 1 - max(sims).item()
            else:
                min_dist = 1.0
            remaining_probes_with_dist.append((probes[i], min_dist))

        top_remaining = sorted(remaining_probes_with_dist, key=lambda x: -x[1])
        
        num_to_add = num_desired - len(selected_probes)
        selected_probes.extend([p[0] for p in top_remaining[:num_to_add]])

    return selected_probes[:num_desired]

def extract_questions(raw_text, num_expected):
    """Extracts clean, complete questions from raw model output."""
    question_pattern = re.compile(r"^\s*(?:[*-]|\d+\.?)?\s*(.*\S.*?\?)\s*$", re.MULTILINE)
    questions = question_pattern.findall(raw_text)
    
    cleaned_questions = []
    for q in questions:
        q = re.sub(r"^(here are|sure, |of course, |certainly, |okay, )", "", q.strip(), flags=re.IGNORECASE)
        if q:
            q = q[0].upper() + q[1:]
        cleaned_questions.append(q)

    seen = set()
    unique_questions = [q for q in cleaned_questions if not (q.lower() in seen or seen.add(q.lower()))]

    if not unique_questions:
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        potential_questions = [re.sub(r"^\s*(?:[*-]|\d+\.?)?\s*", "", line).strip() for line in lines]
        return [q for q in potential_questions if len(q.split()) > 3 and not q.endswith(':')][:num_expected]

    return unique_questions[:num_expected]

def generate_probes(base_prompt, model_name, num_probes, max_attempts=3):
    """Generate diverse probing questions for a prompt using a probe LLM."""
    rich_prompt_template = (
        "You are a probing tool. Your task is to generate questions based on the user's input. "
        "Do not answer the question, but generate more questions. "
        "Based on the following topic:\n--- TOPIC START ---\n{Question}\n--- TOPIC END ---\n"
        f"Generate {max(2 * num_probes, num_probes + 3)} diverse, answerable, and complete questions related to this topic. "
        "Each question must be on a new line, be a full sentence, and end with a question mark. "
        "Do not add any conversational text, prefixes, or introductions."
    )
    minimal_prompt_template = f"{{Question}}\nGenerate {num_probes} different questions."

    for attempt in range(max_attempts):
        prompt_template_str = rich_prompt_template if attempt == 0 else minimal_prompt_template
        prompt_template = PromptTemplate(input_variables=["Question"], template=prompt_template_str)
        
        llm = _get_llm_instance(model_name)
        chain = prompt_template | llm
        
        input_text = truncate_prompt_to_fit(prompt_template_str, base_prompt, config.CONTEXT_LENGTH)
        raw_output = chain.invoke({"Question": input_text})
        
        questions = extract_questions(raw_output, num_probes)
        diverse = diverse_probes(questions, num_probes)
        if len(diverse) >= num_probes:
            return diverse[:num_probes]
            
    if 'diverse' in locals() and len(diverse) >= 2:
        return diverse[:num_probes]
    
    print(f"    [Warning] Probe generation failed for '{base_prompt[:50]}...'. Using fallback.")
    return [f"What is the definition of {base_prompt}?", f"How does {base_prompt} work?"][:num_probes]


def evaluate_answer_accuracy(model_answer, reference_answer):
    """Evaluate semantic similarity between a model's answer and a reference answer."""
    if not model_answer or not reference_answer:
        return 0.0
    
    model_answer_clean = preprocess_text(model_answer)
    reference_answer_clean = preprocess_text(reference_answer)
    
    if not model_answer_clean or not reference_answer_clean:
        return 0.0
    if model_answer_clean == reference_answer_clean:
        return 1.0
        
    try:
        model_emb = get_text_embeddings([model_answer_clean])
        gt_emb = get_text_embeddings([reference_answer_clean])
        
        if model_emb.size == 0 or gt_emb.size == 0: return 0.0
            
        similarity = util.pytorch_cos_sim(model_emb[0], gt_emb[0]).item()
        normalized_similarity = (similarity + 1) / 2
        
        return 0.0 if normalized_similarity < 0.1 else min(1.0, max(0.0, normalized_similarity))
        
    except Exception as e:
        print(f"Error in evaluate_answer_accuracy: {e}. Falling back to Jaccard similarity.")
        model_words = set(model_answer.lower().split())
        ref_words = set(reference_answer.lower().split())
        if not ref_words: return 0.0
        
        intersection = len(model_words.intersection(ref_words))
        union = len(model_words.union(ref_words))
        return intersection / union if union > 0 else 0.0

def compute_robustness_metric(answers):
    """Calculate robustness score based on the consistency of answers."""
    if len(answers) < 2:
        return 0, [], 0, 0, 0, []
        
    embeddings = get_text_embeddings(answers)
    similarities = []
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            similarities.append(sim)
            pairs.append((i, j, sim))
            
    mean_similarity = np.mean(similarities) if similarities else 0
    variance = np.var(similarities) if similarities else 0
    min_sim = np.min(similarities) if similarities else 0
    max_sim = np.max(similarities) if similarities else 0
    
    robustness_score = round(mean_similarity * 100)
    return robustness_score, similarities, variance, min_sim, max_sim, pairs


def compute_domain_robustness_score(domain_prompts, ground_truth_answers, model_name, probe_llms, num_probes):
    """
    Evaluate domain robustness by generating probes for each prompt, getting answers,
    and measuring the consistency of those answers.
    """
    all_prompt_scores = []
    all_prompt_accuracies = []
    prompt_details = []
    
    print(f"[INFO] Starting domain robustness evaluation for {len(domain_prompts)} prompts...")
    for i, (prompt, gt_answer) in enumerate(zip(domain_prompts, ground_truth_answers), 1):
        print(f"\n[Prompt {i}/{len(domain_prompts)}] {prompt}")
        print(f"Ground Truth: {gt_answer}")
        
        model_answer = get_robust_model_answer(prompt, model_name)
        model_answer_accuracy = evaluate_answer_accuracy(model_answer, gt_answer)
        print(f"Answer: {model_answer}")
        print(f"Answer Accuracy: {model_answer_accuracy:.2f}")
        
        prompt_scores = []
        prompt_accuracies = []
        probe_results_for_prompt = []

        for j, probe_llm in enumerate(probe_llms, 1):
            print(f"  [Probe LLM {j}/{len(probe_llms)}] Using: {probe_llm}")
            probes = generate_probes(prompt, probe_llm, num_probes)
            if len(probes) < 2:
                print("    [Skipped] Not enough probes generated.")
                continue

            probe_run_results = []
            for k, p in enumerate(probes, 1):
                print(f"\n    Probe {k}/{len(probes)}: {p}")
                answer = get_robust_model_answer(p, model_name)
                print(f"    Answer: {answer}")
                if answer != "No answer available.":
                    accuracy = evaluate_answer_accuracy(answer, gt_answer)
                    print(f"    Accuracy: {accuracy:.2f}")
                    probe_run_results.append({'probe': p, 'answer': answer, 'accuracy': accuracy})
            
            def is_generic(ans_text):
                return any(gen.lower() in ans_text.strip().lower() for gen in config.GENERIC_ANSWERS)
            
            non_generic_results = [res for res in probe_run_results if not is_generic(res['answer'])]

            if len(non_generic_results) < 2:
                print("    [Skipped] Not enough unique, non-generic answers (after filtering generic).")
                continue

            answers_for_dedup = [r['answer'] for r in non_generic_results]
            embeddings = get_text_embeddings(answers_for_dedup)
            
            final_results = []
            kept_indices = []
            for res_idx, emb in enumerate(embeddings):
                is_dup = False
                for kept_idx in kept_indices:
                    sim = util.pytorch_cos_sim(emb, embeddings[kept_idx]).item()
                    if sim > 0.92:
                        is_dup = True
                        break
                if not is_dup:
                    kept_indices.append(res_idx)
                    final_results.append(non_generic_results[res_idx])

            if len(final_results) < 2:
                print("    [Skipped] Not enough unique, non-generic answers (after filtering duplicates).")
                continue
            
            cleaned_answers = [r['answer'] for r in final_results]
            cleaned_probes = [r['probe'] for r in final_results]
            accuracies = [r['accuracy'] for r in final_results]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0

            robustness_score, _, variance, min_sim, max_sim, _ = compute_robustness_metric(cleaned_answers)
            prompt_scores.append(robustness_score)
            prompt_accuracies.append(avg_accuracy)

            probe_results_for_prompt.append({
                'probe_llm': probe_llm,
                'robustness_score': robustness_score,
                'variance': variance,
                'min_sim': min_sim,
                'max_sim': max_sim,
                'probes': cleaned_probes,
                'answers': cleaned_answers,
                'avg_accuracy': avg_accuracy,
                'accuracies': accuracies,
            })

        if prompt_scores:
            all_prompt_scores.extend(prompt_scores)
            prompt_avg_accuracy = np.mean(prompt_accuracies) if prompt_accuracies else 0
            all_prompt_accuracies.append(prompt_avg_accuracy)
            prompt_details.append({
                'prompt': prompt,
                'ground_truth': gt_answer,
                'model_answer': model_answer,
                'model_answer_accuracy': model_answer_accuracy,
                'scores': prompt_scores,
                'avg_accuracy': prompt_avg_accuracy,
                'probe_results': probe_results_for_prompt
            })
            
    return {
        'domain_mean': np.mean(all_prompt_scores) if all_prompt_scores else 0,
        'domain_median': np.median(all_prompt_scores) if all_prompt_scores else 0,
        'domain_min': np.min(all_prompt_scores) if all_prompt_scores else 0,
        'domain_max': np.max(all_prompt_scores) if all_prompt_scores else 0,
        'domain_avg_accuracy': np.mean(all_prompt_accuracies) if all_prompt_accuracies else 0,
        'all_scores': all_prompt_scores,
        'all_accuracies': all_prompt_accuracies,
        'prompt_results': prompt_details
    }

def print_results(results):
    print("\n=================\nDomain-level Robustness Results:")
    print(f"All Robustness Scores: {results['all_scores']}")
    print(f"Domain Robustness -> Mean: {results['domain_mean']:.2f}, Median: {results['domain_median']:.2f}, Min: {results['domain_min']}, Max: {results['domain_max']}")
    print(f"Domain Average Accuracy -> {results['domain_avg_accuracy']:.2f}")
    
    for pr in results['prompt_results']:
        print(f"\nPrompt: {pr['prompt']}")
        print(f"  Robustness Scores: {pr['scores']}")
        print(f"  Average Accuracy: {pr['avg_accuracy']:.2f}")
        for res in pr['probe_results']:
            print(f"    - Probe LLM: {res['probe_llm']}")
            print(f"      Robustness Score: {res['robustness_score']}, Variance: {res['variance']:.4f}")
            print(f"      Average Accuracy: {res['avg_accuracy']:.2f}")

def save_results_to_csv(results, model_name):
    """Saves the detailed evaluation results to a CSV file, merging repetitive cells for readability."""
    safe_model_name = model_name.replace('/', '_')
    filename = f"evaluation_results_{safe_model_name}.csv"
    
    report_data = []
    
    column_order = [
        'Model Name', 'Original Question', 'GT Answer', 'Model Answer (MA)', 'MA Accuracy',
        'Probe LLM', 'Probe Question', 'Probe Answer', 'Probe Accuracy',
        'Robustness Score', 'Avg Accuracy'
    ]
    
    is_first_row_in_file = True
    for prompt_res in results['prompt_results']:
        is_first_row_for_prompt = True
        for probe_res in prompt_res['probe_results']:
            is_first_row_for_probe_llm = True
            for i in range(len(probe_res['probes'])):
                
                row = {key: '' for key in column_order}

                if is_first_row_for_prompt:
                    row['Model Name'] = model_name if is_first_row_in_file else ""
                    row['Original Question'] = prompt_res['prompt']
                    row['GT Answer'] = prompt_res['ground_truth']
                    row['Model Answer (MA)'] = prompt_res['model_answer']
                    row['MA Accuracy'] = prompt_res['model_answer_accuracy']

                if is_first_row_for_probe_llm:
                    row['Probe LLM'] = probe_res['probe_llm']
                    row['Robustness Score'] = probe_res['robustness_score']
                    row['Avg Accuracy'] = probe_res['avg_accuracy']

                row['Probe Question'] = probe_res['probes'][i]
                row['Probe Answer'] = probe_res['answers'][i]
                row['Probe Accuracy'] = probe_res['accuracies'][i]
                
                report_data.append(row)
                
                is_first_row_in_file = False
                is_first_row_for_prompt = False
                is_first_row_for_probe_llm = False
    
    if not report_data:
        print("\n[WARNING] No data to save to CSV.")
        return
        
    try:
        df = pd.DataFrame(report_data, columns=column_order)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save results to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness.")
    parser.add_argument(
        '--data_source',
        type=str,
        default='poc',
        choices=['medical_qa', 'poc'],
        help='Select the data source for evaluation.'
    )
    args = parser.parse_args()

    model_to_audit = config.MODEL_1_NAME
    print(f"Auditing Model: {model_to_audit}")
    
    questions, ground_truth_answers = load_medical_questions(source=args.data_source, num_questions=config.NUM_QUESTIONS)
    if not questions:
        print("Error: No questions loaded. Exiting.")
        return

    print(f"\nLoaded {len(questions)} questions for evaluation.")

    results = compute_domain_robustness_score(
        domain_prompts=questions,
        ground_truth_answers=ground_truth_answers,
        model_name=model_to_audit,
        probe_llms=config.PROBE_LLM_NAMES,
        num_probes=config.NUM_PROBES
    )

    if results and results['prompt_results']:
        print_results(results)
        save_results_to_csv(results, model_to_audit)

if __name__ == "__main__":
    main()