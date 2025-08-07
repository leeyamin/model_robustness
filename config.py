from dataclasses import dataclass, field
from typing import List
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    audited_llm: str = field(
        metadata={"help": "Name of the main model to be evaluated for robustness"}
    )
    probe_llms: List[str] = field(
        metadata={"help": "List of LLM models used for generating probe questions"}
    )
    embedding_llm: str = field(
        metadata={"help": "Model used for computing semantic similarities between texts"}
    )

@dataclass
class ParametersConfig:
    num_probes: int = field(
        metadata={"help": "Number of probe questions to generate per LLM"}
    )
    context_length: int = field(
        metadata={"help": "Maximum context length for model inputs"}
    )
    num_questions: int = field(
        metadata={"help": "Number of questions to process from the dataset"}
    )
    probe_similarity_threshold: float = field(
        metadata={"help": "Minimum similarity threshold for accepting generated probes"}
    )

@dataclass
class PathsConfig:
    """Configuration for file paths."""
    
    medical_qa_path: str = field(
        metadata={"help": "Path to the processed medical QA dataset CSV file"}
    )

@dataclass
class GenericAnswersConfig:
    answers: List[str] = field(
        metadata={"help": "List of generic answer patterns to filter out from evaluation"}
    )

@dataclass
class EnvironmentConfig:
    hf_hub_disable_progress_bars: bool = field(
        metadata={"help": "Whether to disable Hugging Face progress bar outputs"}
    )

@dataclass
class Config:
    model: ModelConfig = field(
        metadata={"help": "Model-related configuration"}
    )
    parameters: ParametersConfig = field(
        metadata={"help": "Runtime parameters for evaluation"}
    )
    paths: PathsConfig = field(
        metadata={"help": "File path configurations"}
    )
    generic_answers: GenericAnswersConfig = field(
        metadata={"help": "Configuration for filtering generic answers"}
    )
    environment: EnvironmentConfig = field(
        metadata={"help": "Environment and logging settings"}
    )

def register_config():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config) 