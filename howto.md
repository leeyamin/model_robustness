## How to set up and run the project

This guide helps you create a clean Python environment and run the robustness audit end‑to‑end.

### 0) Assumptions
- The processed dataset exists at `data/processed/medical_qa/medical_qa_processed.csv`.
- You have cloned/downloaded the repository and are running commands from the repo root.

If the CSV does not exist, the code will attempt to generate it using `data/processors/medical_qa_processor.py`.

Dataset format and quick-start sample:
- A small illustrative sample with 5 rows is provided at `data/processed/medical_qa/medical_qa_processed_copy.csv` to clarify the expected format and enable a quick run.
- Required CSV columns:
  - `question`: the user question
  - `concise_answer`: the reference/ground-truth answer used for accuracy scoring
  - `original_answer`: optional long-form answer (not used by the pipeline but included for context)

Quick start with the 5-sample file (limits processing to 5): (run this after completing Steps 1–3 (environment setup and dependency install))

```bash
python model_robustness.py hydra.job.chdir=false \
  domain.data_path="data/processed/medical_qa/medical_qa_processed_copy.csv" \
  parameters.num_questions=5
```

### 1) Prerequisites
- **Python**: 3.10 or 3.11 recommended
- **Disk space**: Several GB for model downloads and caches

### 2) Create and activate a virtual environment
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

This will install the libraries used for generation (`langchain`, `ctransformers`), evaluation (`ragas`, `sentence-transformers`, `transformers`), and utilities.

### 4) Run the robustness audit
From the repository root, run the main entrypoint (Hydra loads `conf/config.yaml` by default):

```bash
python model_robustness.py hydra.job.chdir=false
```

On first run, the following resources will be downloaded automatically if not cached:
- Audited LLM and probe LLMs from the `model` section in `conf/config.yaml` (default audited model: `TheBloke/Llama-2-7B-Chat-GGML`)
- Sentence embedding model `paraphrase-MiniLM-L6-v2`
- Evaluator model `google/flan-t5-base` (used by Ragas `AnswerAccuracy`)

Results will be printed to the console and saved in the repo root as CSV files:
- `robustness_summary_<model>.csv`
- `robustness_detailed_<model>.csv`

### 5) Common configuration overrides (CLI)
You can change any setting in `conf/config.yaml` from the command line. Examples:

```bash
# Evaluate less questions (e.g., first 2 rows in the dataset)
python model_robustness.py hydra.job.chdir=false parameters.num_questions=2

# Generate more probes per question (e.g., 5 per probe LLM)
python model_robustness.py hydra.job.chdir=false parameters.num_probes=5

# Evaluate a different audited model
python model_robustness.py hydra.job.chdir=false model.audited_llm=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

# Point to a different processed dataset path
python model_robustness.py hydra.job.chdir=false domain.data_path="data/processed/medical_qa/medical_qa_processed.csv"
```

Hydra composes configs at runtime; overrides do not modify the file on disk.

Note: Passing `hydra.job.chdir=false` keeps the working directory as the repo root so relative paths like `data/...` resolve correctly and CSV outputs are saved in the project root. If you omit it, Hydra will change into an `outputs/...` folder.

### 6) Troubleshooting
- **Slow or large downloads**: Switch to lighter models (see above). First run may take time.
- **High CPU usage**: `ctransformers` runs on CPU by default and uses all cores. Close other heavy apps or consider smaller models.
- **Out of memory / long runs**: Lower `parameters.num_questions` and/or `parameters.num_probes`.
- **No results files**: Check that the dataset path in `conf/config.yaml` (or your CLI override) points to an existing CSV.

### 7) Reproducibility
The script sets deterministic seeds via `utils.set_seeds()` and disables noisy progress bars. You can rerun with the same config to compare results.

### 8) Project entry points and key files
- Main script: `model_robustness.py`
- Config: `conf/config.yaml` (Hydra; override via CLI)
- Processed data: `data/processed/medical_qa/medical_qa_processed.csv`
- Output: CSV summaries saved in the project root

That's it -- with the environment set up and the dataset in place, one command (`python model_robustness.py`) will run the full robustness audit.


