import logging
import os
import re
import subprocess
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import hydra
import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer, util

from config import register_config
import utils

register_config()

load_dotenv()
nltk.download('punkt', quiet=True)

_embedding_model_cache = {}
_llm_instance_cache = {}


def load_domain_questions(cfg: DictConfig, num_questions=None):
    """Load domain-specific questions and answers from the configured data path."""
    if num_questions is None:
        num_questions = cfg.parameters.num_questions
        
    domain_name = cfg.domain.name
    data_path = cfg.domain.data_path
    processor_script = cfg.domain.processor_script
    
    if not os.path.exists(data_path):
        print(f"{domain_name.title()} QA CSV not found at {data_path}. Running {processor_script}...")
        subprocess.run(["python", processor_script], check=True)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Failed to generate {data_path}")

    df = pd.read_csv(data_path, nrows=num_questions)
    questions = df['question'].tolist()
    ground_truth_answers = df['concise_answer'].tolist()
    return questions, ground_truth_answers


def get_text_embeddings(texts, cfg: DictConfig):
    """Generate sentence embeddings."""
    model_name = cfg.model.embedding_llm
    
    if model_name not in _embedding_model_cache:
        _embedding_model_cache[model_name] = SentenceTransformer(model_name)
    
    model = _embedding_model_cache[model_name]
    return model.encode(texts)


def _get_llm_instance(model_name, cfg: DictConfig):
    """Initialize and return a model instance for text generation."""
    cache_key = (model_name, cfg.parameters.context_length)
    
    if cache_key not in _llm_instance_cache:
        config_dict = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'context_length': cfg.parameters.context_length,
            'stop': ["\nQ:", "\nQuestion:", "Question:"]
        }
        kwargs = {'config': config_dict, 'threads': os.cpu_count() or 1}
        _llm_instance_cache[cache_key] = CTransformers(model=model_name, **kwargs)
    
    return _llm_instance_cache[cache_key]

def _create_domain_prompt_template(domain="medical"):
    """Create domain-specific prompt template."""
    templates = {
        "medical": (
            "### INSTRUCTION:\nYou are a medical expert. Provide a direct, concise, and factual answer to the user's question. "
            "Do not ask for more details, do not ask clarifying questions, and do not greet the user. "
            "Your answer should be three sentences or less.\n\n"
            "### QUESTION:\n{Question}\n\n"
            "### ANSWER:\n"
        ),
    }
    template_str = templates.get(domain, templates["medical"])  # Use medical as default
    return PromptTemplate(input_variables=["Question"], template=template_str)


def _clean_model_response(raw_answer):
    """Clean and format model response by removing artifacts and filler patterns."""
    answer = raw_answer.strip()
    
    answer = re.split(r'\\n\\s*(Q|A|Question|Answer):', answer)[0].strip()
    
    filler_patterns = [
        r"^(Sure|Of course|Certainly),? here (is|'s) (the|a concise) answer:?",
        r"^Here is a concise and factual answer:?",
        r"^(Here's|Here is) the answer:?",
        r"^(Answer|A)\\s*[:：]?",
        r"^Answer concisely\\s*[:.]?"
    ]
    
    for pattern in filler_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).lstrip(' :.\\n').strip()
    
    return answer


def get_model_answer(prompt, model_name, cfg: DictConfig, max_attempts=3):
    """Generate a clean answer from a model with retries."""
    prompt_template = _create_domain_prompt_template(cfg.domain.name)
    llm = _get_llm_instance(model_name, cfg)
    chain = prompt_template | llm
    
    truncated_prompt = utils.truncate_prompt_to_fit(prompt_template.template, prompt, cfg.parameters.context_length)
    if not truncated_prompt and prompt:
        return "Prompt was too long for context and could not be processed."

    for _ in range(max_attempts):
        raw_answer = chain.invoke({"Question": truncated_prompt})
        if isinstance(raw_answer, dict) and 'text' in raw_answer:
            raw_answer = raw_answer['text']
        
        answer = _clean_model_response(raw_answer)
        
        full_prompt_sent = prompt_template.format(Question=truncated_prompt)
        if answer.lower().startswith(full_prompt_sent.lower().strip()):
            answer = answer[len(full_prompt_sent):].strip()
        elif answer.lower().startswith(truncated_prompt.lower().strip()):
            answer = answer[len(truncated_prompt):].strip()

        if answer:
            return answer
    
    return "No answer available."

def get_robust_model_answer(prompt, model_name, cfg: DictConfig, max_retries=3):
    """Get reliable model responses by retrying on failures."""
    for i in range(max_retries):
        answer = get_model_answer(prompt, model_name, cfg)
        
        if answer and answer.strip() and answer != "No answer available.":
            return answer
            
        failure_reason = "empty response" if not answer or not answer.strip() else "no answer available"
        if i < max_retries - 1:
            print(f"    [INFO] {failure_reason}. Retrying ({i+1}/{max_retries-1})...")
    
    return "No answer available."

def _create_probe_templates(domain, num_probes):
    """Create domain-specific probe generation templates."""
    target_count = max(2 * num_probes, num_probes + 3)
    
    templates = {
        "medical": {
            "rich": (
                "You are an expert in generating patient questions. "
                "Rephrase and expand the user's question to generate diverse, contextually similar questions that a patient might ask a doctor. "
                "Each question must be complete and standalone, including the main topic.\n\n"
                "--- MAIN TOPIC ---\n{Question}\n--- END MAIN TOPIC ---\n\n"
                f"Generate {target_count} patient-like questions about the main topic. "
                "Each question must be on a new line and end with a question mark. "
                "Do not add conversational text, prefixes, or introductions."
            ),
            "minimal": f"Generate {num_probes} different medical questions about {{Question}} from a patient's perspective."
        },
    }
    
    domain_templates = templates.get(domain, templates["medical"])
    return domain_templates["rich"], domain_templates["minimal"]


def generate_probes(base_prompt, model_name, num_probes, cfg: DictConfig, max_attempts=3):
    """Generate diverse probing questions using domain-specific templates."""
    rich_template, minimal_template = _create_probe_templates(cfg.domain.name, num_probes)
    
    llm = _get_llm_instance(model_name, cfg)
    
    for attempt in range(max_attempts):
        template_str = rich_template if attempt == 0 else minimal_template
        prompt_template = PromptTemplate(input_variables=["Question"], template=template_str)
        chain = prompt_template | llm
        
        input_text = utils.truncate_prompt_to_fit(template_str, base_prompt, cfg.parameters.context_length)
        raw_output = chain.invoke({"Question": input_text})
        
        questions = utils.extract_questions(raw_output, num_probes)
        if len(questions) >= num_probes:
            return questions[:num_probes]
    
    print(f"    [ERROR] Probe generation failed for '{base_prompt[:50]}...' after {max_attempts} attempts.")
    return []

def _validate_similarity_inputs(text1, text2):
    """Validate inputs for similarity computation."""
    if not text1 or not text2:
        return False, 0.0
    
    text1_clean = utils.preprocess_text(text1)
    text2_clean = utils.preprocess_text(text2)
    
    if not text1_clean or not text2_clean:
        return False, 0.0
    if text1_clean == text2_clean:
        return False, 1.0
    
    return True, (text1_clean, text2_clean)


def _compute_cosine_similarity(text1_clean, text2_clean, cfg):
    """Compute raw cosine similarity between preprocessed texts."""
    text1_emb = get_text_embeddings([text1_clean], cfg)
    text2_emb = get_text_embeddings([text2_clean], cfg)
    
    if text1_emb.size == 0 or text2_emb.size == 0:
        return None
    
    return util.pytorch_cos_sim(text1_emb[0], text2_emb[0]).item()


def _normalize_similarity_score(raw_similarity):
    """Normalize cosine similarity to 0-1 range with threshold."""
    normalized = (raw_similarity + 1) / 2
    return max(0.0, min(1.0, normalized))


def compute_semantic_similarity(text1, text2, cfg: DictConfig):
    """Compute semantic similarity between two texts."""
    should_continue, result = _validate_similarity_inputs(text1, text2)
    if not should_continue:
        return result
    
    text1_clean, text2_clean = result
    
    raw_similarity = _compute_cosine_similarity(text1_clean, text2_clean, cfg)
    if raw_similarity is None:
        return 0.0

    return _normalize_similarity_score(raw_similarity)

def compute_robustness_metric(answers, cfg: DictConfig):
    """Calculate robustness score based on answer consistency."""
    if len(answers) < 2:
        return 0, []
        
    embeddings = get_text_embeddings(answers, cfg)
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
            similarities.append(sim)
    
    mean_similarity = np.mean(similarities)
    robustness_score = round(mean_similarity * 100)
    return robustness_score, similarities

def _validate_probes(probes, original_prompt, cfg):
    """Validate probes for semantic similarity to original question."""
    validated_probes = []
    for p in probes:
        similarity = compute_semantic_similarity(p, original_prompt, cfg)
        if cfg.parameters.probe_similarity_threshold <= similarity <= 0.99:
            validated_probes.append(p)
        elif similarity < cfg.parameters.probe_similarity_threshold:
            print(f"    [Skipped Probe] Low similarity to original question ({similarity:.2f} < {cfg.parameters.probe_similarity_threshold}): {p}")
        else:
            print(f"    [Skipped Probe] Too similar to original question ({similarity:.2f} > 0.99): {p}")
    return validated_probes


def _is_generic_answer(ans_text, cfg, threshold=0.8):
    """Enhanced generic detection using both pattern matching and semantic similarity."""
    answer_clean = ans_text.strip().lower()
    
    if any(gen.lower() in answer_clean for gen in list(cfg.generic_answers.answers)):
        return True
    
    generic_patterns = [
        r"^(i|i'm) (don't know|not sure|unsure|uncertain)",
        r"^(sorry|apologize).{0,20}(cannot|can't|unable)",
        r"^(that's|that is) a (good|great|interesting) question",
        r"^i (would need|need) more (information|details)",
        r"^(please|you should) consult (a|your) (doctor|physician|medical professional)",
        r"^(i|i'm) not (qualified|able|authorized) to"
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, answer_clean):
            return True
    
    answer_embedding = get_text_embeddings([ans_text], cfg)[0]
    generic_embeddings = get_text_embeddings(list(cfg.generic_answers.answers), cfg)

    for generic_emb in generic_embeddings:
        similarity = util.pytorch_cos_sim(answer_embedding, generic_emb).item()
        if similarity > threshold:
            return True
    
    return False


def _generate_probe_answers(validated_probes, model_name, gt_answer, cfg):
    """Generate answers for validated probes and compute accuracies."""
    probe_run_results = []
    for k, p in enumerate(validated_probes, 1):
        print(f"\n    Probe {k}/{len(validated_probes)}: {p}")
        answer = get_robust_model_answer(p, model_name, cfg)
        print(f"    Answer: {answer}")
        if answer != "No answer available.":
            accuracy = compute_semantic_similarity(answer, gt_answer, cfg)
            print(f"    Accuracy: {accuracy:.2f}")
            probe_run_results.append({'probe': p, 'answer': answer, 'accuracy': accuracy})
    return probe_run_results


def _filter_generic_answers(probe_run_results, cfg):
    """Filter out generic/uninformative answers."""
    non_generic_results = [res for res in probe_run_results if not _is_generic_answer(res['answer'], cfg)]
    return non_generic_results


def _process_single_probe_llm(prompt, gt_answer, probe_llm, model_name, num_probes, cfg):
    """Process probes for a single probe LLM and return results."""
    print(f"  [Probe LLM] Using: {probe_llm}")
    
    probes = generate_probes(prompt, probe_llm, num_probes, cfg)
    validated_probes = _validate_probes(probes, prompt, cfg)
    
    if len(validated_probes) < 2:
        print("    [Skipped] Not enough validated probes generated after filtering.")
        return None
    
    probe_run_results = _generate_probe_answers(validated_probes, model_name, gt_answer, cfg)
    
    final_results = _filter_generic_answers(probe_run_results, cfg)
    
    if len(final_results) < 2:
        print("    [Skipped] Not enough unique, non-generic answers (after filtering).")
        return None
    
    cleaned_answers = [r['answer'] for r in final_results]
    cleaned_probes = [r['probe'] for r in final_results]
    accuracies = [r['accuracy'] for r in final_results]
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    robustness_score, _ = compute_robustness_metric(cleaned_answers, cfg)
    
    return {
        'probe_llm': probe_llm,
        'robustness_score': robustness_score,
        'probes': cleaned_probes,
        'answers': cleaned_answers,
        'avg_accuracy': avg_accuracy,
        'accuracies': accuracies,
    }


def _process_single_prompt(prompt, gt_answer, probe_llms, model_name, num_probes, cfg, prompt_index, total_prompts):
    """Process all probe LLMs for a single prompt."""
    print(f"\n[Prompt {prompt_index}/{total_prompts}] {prompt}")
    print(f"Ground Truth: {gt_answer}")
    
    prompt_scores = []
    prompt_accuracies = []
    probe_results_for_prompt = []
    
    for j, probe_llm in enumerate(probe_llms, 1):
        result = _process_single_probe_llm(prompt, gt_answer, probe_llm, model_name, num_probes, cfg)
        if result:
            prompt_scores.append(result['robustness_score'])
            prompt_accuracies.append(result['avg_accuracy'])
            probe_results_for_prompt.append(result)
    
    if not prompt_scores:
        return None
    
    return {
        'prompt': prompt,
        'ground_truth': gt_answer,
        'scores': prompt_scores,
        'avg_accuracy': np.mean(prompt_accuracies),
        'probe_results': probe_results_for_prompt
    }


def compute_domain_robustness_score(domain_prompts, ground_truth_answers, model_name, probe_llms, num_probes, cfg: DictConfig):
    """Evaluate domain robustness by testing consistency across probe questions."""
    print(f"[INFO] Starting domain robustness evaluation for {len(domain_prompts)} prompts...")
    
    all_prompt_scores = []
    all_prompt_accuracies = []
    prompt_details = []
    
    for i, (prompt, gt_answer) in enumerate(zip(domain_prompts, ground_truth_answers), 1):
        prompt_result = _process_single_prompt(prompt, gt_answer, probe_llms, model_name, num_probes, cfg, i, len(domain_prompts))
        all_prompt_scores.extend(prompt_result['scores'])
        all_prompt_accuracies.append(prompt_result['avg_accuracy'])
        prompt_details.append(prompt_result)
    
    return {
        'overall_robustness': np.mean(all_prompt_scores) if all_prompt_scores else 0,
        'overall_accuracy': np.mean(all_prompt_accuracies) if all_prompt_accuracies else 0,
        'total_questions': len(prompt_details),
        'prompt_results': prompt_details
    }


def save_results_to_csv(results, model_name):
    """Save both summary and detailed results to separate csv files."""
    utils.save_summary_results(results, model_name)
    utils.save_detailed_results(results, model_name)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    utils.set_seeds()

    model_to_audit = cfg.model.audited_llm
    print(f"Auditing Model: {model_to_audit}")
    
    questions, ground_truth_answers = load_domain_questions(cfg, cfg.parameters.num_questions)
    print(f"\nLoaded {len(questions)} questions for evaluation.")

    results = compute_domain_robustness_score(
        domain_prompts=questions,
        ground_truth_answers=ground_truth_answers,
        model_name=model_to_audit,
        probe_llms=cfg.model.probe_llms,
        num_probes=cfg.parameters.num_probes,
        cfg=cfg
    )

    utils.print_results(results, model_to_audit)
    save_results_to_csv(results, model_to_audit)

if __name__ == "__main__":
    main()