from dataclasses import dataclass, field
from typing import List
from typing import Dict
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    audited_llm: str = field(
        metadata={"help": "Name of the main model to be evaluated for robustness"}
    )
    probe_llms: List[str] = field(
        metadata={"help": "List of LLM models used for generating probe questions"}
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
class DomainTemplatesConfig:
    rich: str = field(
        metadata={"help": "Rich prompt template for probe generation (may contain {Question} and {target_count})"}
    )
    minimal: str = field(
        metadata={"help": "Minimal prompt template for probe generation (may contain {Question} and {num_probes})"}
    )

@dataclass
class DomainConfig:
    name: str = field(
        metadata={"help": "Domain name (e.g., medical)"}
    )
    processor_script: str = field(
        metadata={"help": "Path to data processor script for the domain"}
    )
    data_path: str = field(
        metadata={"help": "Path to processed dataset for the domain"}
    )
    probe_templates: DomainTemplatesConfig = field(
        metadata={"help": "Probe generation templates for the domain"}
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
    domain: DomainConfig = field(
        metadata={"help": "Domain configuration including probe templates"}
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