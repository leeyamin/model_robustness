import random
import re
import numpy as np
import pandas as pd
import torch


def set_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_tokens(text):
    """Count tokens in text."""
    return len(text.split())


def truncate_prompt_to_fit(prompt_template_str, user_input, max_context_tokens, placeholder_key="Question"):
    """Truncate user input to fit within the model context window."""
    dummy_input = "PLACEHOLDER"
    template_vars = {placeholder_key: dummy_input}
    prompt_with_dummy = prompt_template_str.format(**template_vars)
    prompt_tokens = count_tokens(prompt_with_dummy) - count_tokens(dummy_input)

    allowed_input_tokens = max_context_tokens - prompt_tokens
    if allowed_input_tokens <= 0:
        print(f"[WARNING] Template requires {prompt_tokens} tokens, exceeds context limit of {max_context_tokens}.")
        return ""

    input_tokens = user_input.split()
    if len(input_tokens) > allowed_input_tokens:
        return ' '.join(input_tokens[:allowed_input_tokens])
    return user_input


def extract_questions(raw_text, num_expected):
    """Extract questions from text, handling various formats and numbering."""
    
    if not raw_text or not raw_text.strip():
        return []
    
    question_pattern = re.compile(r"^\s*(?:[*-]|\d+\.?)?\s*(.+\?)\s*$", re.MULTILINE)
    questions = question_pattern.findall(raw_text)
    
    if questions:
        cleaned_questions = [_clean_question(q) for q in questions if _clean_question(q)]
        unique_questions = _remove_duplicates(cleaned_questions)
    else:
        unique_questions = _extract_fallback_questions(raw_text)
    
    return unique_questions[:num_expected]


def _clean_question(question):
    """Clean and normalize a single question."""
    q = question.strip()
    
    prefixes = ["here are", "sure,", "of course,", "certainly,", "okay,"]
    for prefix in prefixes:
        if q.lower().startswith(prefix):
            q = q[len(prefix):].strip()
            break
    
    if q:
        q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()
    
    return q


def _remove_duplicates(questions):
    """Remove duplicate questions while preserving order."""
    seen = set()
    unique = []
    for q in questions:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)
    return unique


def _extract_fallback_questions(raw_text):
    """Extract questions when primary pattern fails."""
    
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    questions = []
    
    for line in lines:
        line = re.sub(r"^\s*(?:[*-]|\d+\.?)\s*", "", line).strip()
        if len(line.split()) > 3 and not line.endswith(':'):
            questions.append(line)
    
    return questions


def print_robustness_summary(results, model_name):
    """Print a high-level robustness evaluation summary."""
    print("\n" + "="*50)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("="*50)
    print(f"Model Evaluated: {model_name}")
    print(f"Overall Robustness Score: {results['overall_robustness']:.2f}")
    print(f"Overall Accuracy Score: {results['overall_accuracy']:.2f}")
    print(f"Total Questions Evaluated: {results['total_questions']}")


def print_detailed_results(results):
    """Print detailed per-prompt robustness results."""
    if not results.get('prompt_results') or any(not pr for pr in results['prompt_results']):
        print("\nNo detailed results available.")
    
    print(f"\n{'-'*50}")
    print("DETAILED RESULTS BY PROMPT")
    print(f"{'-'*50}")
    
    for i, pr in enumerate(results['prompt_results'], 1):
        print(f"\n[{i}] Prompt: {pr['prompt']}")
        print(f"    Robustness Scores: {pr['scores']}")
        print(f"    Average Accuracy: {pr['avg_accuracy']:.2f}")
        
        for j, res in enumerate(pr['probe_results'], 1):
            print(f"    [{i}.{j}] Probe LLM: {res['probe_llm']}")
            print(f"          Robustness Score: {res['robustness_score']}")
            print(f"          Average Accuracy: {res['avg_accuracy']:.2f}")


def print_results(results, model_name):
    """Print complete robustness evaluation results."""
    print_robustness_summary(results, model_name)
    print_detailed_results(results)


def save_summary_results(results, model_name):
    """Save a concise summary with model performance metrics."""
    safe_model_name = model_name.replace('/', '_')
    filename = f"robustness_summary_{safe_model_name}.csv"
    
    summary_data = []
    summary_data.append({
        'Model Name': model_name,
        'Overall Robustness Score': f"{results['overall_robustness']:.2f}",
        'Overall Accuracy Score': f"{results['overall_accuracy']:.2f}",
        'Total Questions Evaluated': results['total_questions']
    })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(filename, index=False)
    print(f"\nSummary results saved to {filename}")


def _create_detailed_row(model_name, prompt_res, probe_res, probe_index, is_first_model, is_first_prompt, is_first_probe_llm):
    """Create a single detailed CSV row with appropriate field population."""
    return {
        'Model Name': model_name if is_first_model else "",
        'Original Question': prompt_res['prompt'] if is_first_prompt else "",
        'Ground Truth Answer': prompt_res['ground_truth'] if is_first_prompt else "",
        'Probe LLM': probe_res['probe_llm'] if is_first_probe_llm else "",
        'Robustness Score': probe_res['robustness_score'] if is_first_probe_llm else "",
        'Probe Question': probe_res['probes'][probe_index],
        'Probe Answer': probe_res['answers'][probe_index],
        'Probe Accuracy': f"{probe_res['accuracies'][probe_index]:.3f}"
    }


def _extract_detailed_data(results, model_name):
    """Extract all detailed data into flat CSV structure."""
    detailed_data = []
    total_row_index = 0
    
    for prompt_index, prompt_res in enumerate(results['prompt_results']):
        for probe_llm_index, probe_res in enumerate(prompt_res['probe_results']):
            for probe_index in range(len(probe_res['probes'])):
                
                is_first_model = total_row_index == 0
                is_first_prompt = probe_llm_index == 0 and probe_index == 0
                is_first_probe_llm = probe_index == 0
                
                row = _create_detailed_row(
                    model_name, prompt_res, probe_res, probe_index,
                    is_first_model, is_first_prompt, is_first_probe_llm
                )
                detailed_data.append(row)
                total_row_index += 1
    
    return detailed_data


def save_detailed_results(results, model_name):
    """Save a comprehensive breakdown of results"""
    safe_model_name = model_name.replace('/', '_')
    filename = f"robustness_detailed_{safe_model_name}.csv"
    
    detailed_data = _extract_detailed_data(results, model_name)
    
    if not detailed_data:
        print("\n[WARNING] No detailed data to save.")
        return
    
    column_order = [
        'Model Name', 'Original Question', 'GT Answer',
        'Probe LLM', 'Probe Question', 'Probe Answer', 'Probe Accuracy', 'Robustness Score'
    ]
    
    df = pd.DataFrame(detailed_data, columns=column_order)
    df.to_csv(filename, index=False)
    print(f"\nDetailed results saved to {filename}")
