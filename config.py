import os

# Model Configurations
MODEL_1_NAME = "TheBloke/Llama-2-7B-Chat-GGML"
PROBE_LLM_NAMES = [
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    "TheBloke/Llama-2-7B-Chat-GGML"
]
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Execution Parameters
NUM_PROBES = 5
CONTEXT_LENGTH = 1024
SEED = 42
NUM_QUESTIONS = 3

# Data paths
MEDICAL_QA_PATH = "data/processed/medical_qa/medical_qa_final.csv"

POC_DATA = {
    "questions": [
        "What are the symptoms of a heart attack?",
        "What is the recommended treatment for type 2 diabetes?",
        "How can one prevent the flu?",
        "What are the side effects of penicillin?",
        "What is the difference between a virus and a bacteria?"
    ],
    "answers": [
        "Symptoms of a heart attack include chest pain, shortness of breath, and discomfort in other areas of the upper body.",
        "Treatment for type 2 diabetes typically includes lifestyle changes like diet and exercise, and may also involve medications such as metformin.",
        "The flu can be prevented by getting an annual flu shot, washing hands frequently, and avoiding close contact with sick people.",
        "Common side effects of penicillin include nausea, diarrhea, and allergic reactions such as rashes.",
        "A virus is a non-living infectious agent that replicates inside living cells, while a bacterium is a single-celled living microorganism."
    ]
}

# Answer Filtering
GENERIC_ANSWERS = [
    "I don't know.",
    "I'm not sure.",
    "As an AI language model, I cannot answer that.",
    "Sorry, I do not have that information.",
    "I'm unable to provide that information.",
    "That's a good question."
]

# Environment setup
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" 