import logging
import os
import re
import subprocess
import warnings

import ragas_wrappers

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

from config import register_config
import utils

register_config()

load_dotenv()
nltk.download('punkt', quiet=True)

_embedding_model_cache = {}
_llm_instance_cache = {}


def load_domain_questions(cfg: DictConfig, num_questions):
    """Load domain-specific questions and answers from the configured data path."""
    domain_name = cfg.domain.name
    data_path = cfg.domain.data_path

    if not os.path.exists(data_path):
        processor_script = cfg.domain.processor_script
        print(f"{domain_name.title()} QA CSV not found at {data_path}. Running {processor_script}...")
        try:
            subprocess.run(["python", processor_script], check=True)
        except Exception as e:
            print(f"Failed to run processor script: {e}")
            return [], []

        if not os.path.exists(data_path):
            print(f"Failed to generate {data_path}")
            return [], []

    df = pd.read_csv(data_path, nrows=num_questions)
    questions = df['question'].tolist()
    gt_answers = df['concise_answer'].tolist()
    return questions, gt_answers


def _get_llm_instance(cfg: DictConfig):
    """Initialize and return a model instance for text generation."""
    cache_key = (cfg.model.audited_llm, cfg.parameters.context_length)
    
    if cache_key not in _llm_instance_cache:
        config_dict = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'context_length': cfg.parameters.context_length,
            'stop': ["\nQ:", "\nQuestion:", "Question:"]
        }
        kwargs = {'config': config_dict, 'threads': os.cpu_count() or 1}
        _llm_instance_cache[cache_key] = CTransformers(model=cfg.model.audited_llm, **kwargs)
    
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


def get_model_answer(prompt, cfg: DictConfig, max_attempts=3):
    """Generate a clean answer from a model with retries."""
    prompt_template = _create_domain_prompt_template(cfg.domain.name)
    llm = _get_llm_instance(cfg)
    chain = prompt_template | llm
    
    truncated_prompt = utils.truncate_prompt_to_fit(prompt_template.template, prompt, cfg.parameters.context_length)
    if not truncated_prompt and prompt:
        return "Prompt was too long for context and could not be processed."

    for _ in range(max_attempts):
        raw_response = chain.invoke({"Question": truncated_prompt})
        if isinstance(raw_response, dict) and 'text' in raw_response:
            raw_response = raw_response['text']
        
        response = _clean_model_response(raw_response)
        
        full_prompt_sent = prompt_template.format(Question=truncated_prompt)
        if response.lower().startswith(full_prompt_sent.lower().strip()):
            response = response[len(full_prompt_sent):].strip()
        elif response.lower().startswith(truncated_prompt.lower().strip()):
            response = response[len(truncated_prompt):].strip()

        if response:
            return response
    
    return ""

def get_robust_model_answer(prompt, cfg: DictConfig, max_retries=3):
    """Get reliable model responses by retrying on failures."""
    for i in range(max_retries):
        response = get_model_answer(prompt, cfg)
        
        if response and response.strip() and response != "":
            return response
            
        if i < max_retries - 1:
            print(f"    [INFO] Failed to generate a response. Retrying ({i+1}/{max_retries-1})...")
    
    return ""

def _create_probe_templates(cfg: DictConfig):
    """Create probe generation templates from domain config."""
    num_probes = cfg.parameters.num_probes
    target_count = max(2 * num_probes, num_probes + 3)

    try:
        rich_template = cfg.domain.probe_templates.rich
        minimal_template = cfg.domain.probe_templates.minimal
    except Exception as exc:
        raise ValueError(
            "Domain probe templates must be defined under domain.probe_templates in the config."
        ) from exc

    # Replace numeric placeholders while keeping {Question} for later formatting
    rich_template = str(rich_template).replace("{target_count}", str(target_count))
    minimal_template = str(minimal_template).replace("{num_probes}", str(num_probes))

    return rich_template, minimal_template


def generate_probes(user_input, cfg: DictConfig, max_attempts=3):
    """Generate diverse probing questions using domain-specific templates."""
    rich_template, minimal_template = _create_probe_templates(cfg)
    
    llm = _get_llm_instance(cfg)

    for attempt in range(max_attempts):
        template_str = rich_template if attempt == 0 else minimal_template
        prompt_template = PromptTemplate(input_variables=["Question"], template=template_str)
        chain = prompt_template | llm
        
        input_text = utils.truncate_prompt_to_fit(template_str, user_input, cfg.parameters.context_length)
        raw_output = chain.invoke({"Question": input_text})

        num_probes = cfg.parameters.num_probes
        probe_questions = utils.extract_questions(raw_output, num_probes)
        if len(probe_questions) >= num_probes:
            return probe_questions[:num_probes]
    
    print(f"    [ERROR] Probe generation failed for '{user_input[:50]}...' after {max_attempts} attempts.")
    return []

def compute_robustness_metric(answers):
    """Calculate the robustness score based on the answers' similarity."""
    if len(answers) < 2:
        return 0

    similarities = []
    for idx, response in enumerate(answers):
        for next_idx in range(idx + 1, len(answers)):
            similarity_score = ragas_wrappers.compute_ragas_semantic_similarity(response=response,
                                                                                reference=answers[next_idx])
            similarities.append(similarity_score)

    mean_similarity = np.mean(similarities)
    robustness_score = round(mean_similarity * 100)

    return robustness_score

def _validate_probes(probe_questions, user_input, cfg):
    """Validate probes for semantic similarity to the original input question."""
    validated_probes = []
    for probe in probe_questions:
        similarity_score = ragas_wrappers.compute_ragas_semantic_similarity(response=probe, reference=user_input)
        if cfg.parameters.probe_similarity_threshold <= similarity_score <= 0.99:
            validated_probes.append(probe)
        elif similarity_score < cfg.parameters.probe_similarity_threshold:
            print(f"    [Skipped Probe] Low similarity to original question ({similarity_score:.2f} < {cfg.parameters.probe_similarity_threshold}): {probe}")
        else:
            print(f"    [Skipped Probe] Too similar to original question ({similarity_score:.2f} > 0.99): {probe}")
    return validated_probes


def _is_generic_answer(response, cfg, threshold=0.8):
    """Enhanced generic detection using both pattern matching and semantic similarity."""
    response = response.strip().lower()
    
    if any(gen.lower() in response for gen in list(cfg.generic_answers.answers)):
        return True
    
    for pattern in cfg.generic_answers.answers:
        if re.search(pattern, response):
            return True

    sentences = re.split(r'(?<=[.!?])\s+', response)
    for sentence in sentences:
        similarity = ragas_wrappers.compute_ragas_semantic_similarity(response, sentence)
        if similarity > threshold:
            return True
    
    return False


def _generate_probe_answers(validated_probes, gt_answer, cfg):
    """Generate answers for validated probes and compute accuracies."""
    probe_run_results = []
    for idx, probe_question in enumerate(validated_probes, 1):
        print(f"\n    Probe {idx}/{len(validated_probes)}: {probe_question}")
        response = get_robust_model_answer(probe_question, cfg)
        print(f"    Response: {response}")
        if response is not None:
            answer_accuracy_score = ragas_wrappers.compute_ragas_answer_accuracy(user_input=probe_question,
                                                                                 response=response,
                                                                                 reference=gt_answer)
            print(f"    Accuracy: {answer_accuracy_score:.2f}")
            probe_run_results.append({'probe': probe_question, 'answer': response, 'accuracy': answer_accuracy_score})
    return probe_run_results


def _filter_generic_answers(probe_run_results, cfg):
    """Filter out generic/uninformative answers."""
    non_generic_results = [res for res in probe_run_results if not _is_generic_answer(res['answer'], cfg)]
    return non_generic_results


def _process_single_probe_llm(user_input, reference, probe_llm, cfg):
    """Process probes for a single probe LLM and return results."""
    print(f"  [Probe LLM] Using: {probe_llm}")
    
    probe_questions = generate_probes(user_input, cfg)

    # poc: filter out probes with low semantic similarity to the original question
    # probe_questions = _validate_probes(probe_questions, user_input, cfg)
    
    if len(probe_questions) < 2:
        print("    [Skipped] Not enough validated probes generated after filtering.")
        return None
    
    probe_run_results = _generate_probe_answers(probe_questions, reference, cfg)

    # poc: filter out generic responses
    # probe_run_results = _filter_generic_answers(probe_run_results, cfg)
    #
    # if len(probe_run_results) < 2:
    #     print("    [Skipped] Not enough unique, non-generic answers (after filtering).")
    #     return None
    
    cleaned_answers = [r['answer'] for r in probe_run_results]
    cleaned_probes = [r['probe'] for r in probe_run_results]
    accuracies = [r['accuracy'] for r in probe_run_results]
    avg_accuracy = np.mean(accuracies) if accuracies else None
    robustness_score = compute_robustness_metric(cleaned_answers)
    
    return {
        'probe_llm': probe_llm,
        'robustness_score': robustness_score,
        'probes': cleaned_probes,
        'answers': cleaned_answers,
        'avg_accuracy': avg_accuracy,
        'accuracies': accuracies,
    }


def _process_single_prompt(user_input, reference, cfg, prompt_index, total_prompts):
    """Process all probe LLMs for a single prompt."""
    print(f"\n[Prompt {prompt_index}/{total_prompts}] {user_input}")
    print(f"Ground Truth: {reference}")
    
    prompt_scores = []
    prompt_accuracies = []
    probe_results_for_prompt = []
    
    for j, probe_llm in enumerate(cfg.model.probe_llms, 1):
        result = _process_single_probe_llm(user_input, reference, probe_llm, cfg)

        if result:
            prompt_scores.append(result['robustness_score'])
            prompt_accuracies.append(result['avg_accuracy'])
            probe_results_for_prompt.append(result)

    return {
        'prompt': user_input,
        'ground_truth': reference,
        'scores': prompt_scores,
        'avg_accuracy': np.mean(prompt_accuracies),
        'probe_results': probe_results_for_prompt
    }
    return None


def compute_domain_robustness_score(domain_questions, gt_answers, cfg: DictConfig):
    """Evaluate domain robustness by testing consistency across probe questions."""
    print(f"[INFO] Starting domain robustness evaluation for {len(domain_questions)} questions...")
    
    all_question_scores = []
    all_question_accuracies = []
    prompt_details = []
    
    for i, (user_input, reference) in enumerate(zip(domain_questions, gt_answers), 1):
        prompt_result = _process_single_prompt(user_input, reference, cfg, i, len(domain_questions))
        all_question_scores.extend(prompt_result['scores'])
        all_question_accuracies.append(prompt_result['avg_accuracy'])
        prompt_details.append(prompt_result)
    
    return {
        'overall_robustness': np.mean(all_question_scores) if all_question_scores else 0,
        'overall_accuracy': np.mean(all_question_accuracies) if all_question_accuracies else 0,
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
    
    questions, gt_answers = load_domain_questions(cfg, cfg.parameters.num_questions)
    print(f"\nLoaded {len(questions)} questions for evaluation.")

    results = compute_domain_robustness_score(
        domain_questions=questions,
        gt_answers=gt_answers,
        cfg=cfg
    )

    utils.print_results(results, model_to_audit)
    save_results_to_csv(results, model_to_audit)

if __name__ == "__main__":
    main()