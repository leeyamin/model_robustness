"""
This file contains wrappers for ragas metrics to be used with local models for the poc purposes.
Eventually, this code is intended to be moved to ragas and the wrapper will be removed, implementing the
functionality directly in ragas using the openai api.

The metrics implemented here are:
SemanticSimilarity
https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/semantic_similarity/

AnswerAccuracy
https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/nvidia_metrics/#answer-accuracy
"""

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy

from sentence_transformers import SentenceTransformer
from typing import List
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
import asyncio


class SentenceTransformerWrapper:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    async def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


async def run_semantic_similarity(embeddings, sample):
    scorer = SemanticSimilarity(embeddings=embeddings)
    semantic_similarity_score = await scorer.single_turn_ascore(sample)

    return semantic_similarity_score


async def run_answer_accuracy(sample):
    model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
    )
    llm = HuggingFacePipeline(pipeline=model)

    evaluator_llm = LangchainLLMWrapper(llm)
    scorer = AnswerAccuracy(llm=evaluator_llm)
    answer_accuracy_score = await scorer.single_turn_ascore(sample)

    return answer_accuracy_score

def compute_ragas_semantic_similarity(response, reference):
    embeddings = SentenceTransformerWrapper()
    sample = SingleTurnSample(response=response, reference=reference)
    semantic_similarity_score = asyncio.run(run_semantic_similarity(embeddings, sample))
    return semantic_similarity_score

def compute_ragas_answer_accuracy(user_input, response, reference):
    sample = SingleTurnSample(user_input=user_input, response=response, reference=reference)
    answer_accuracy_score = asyncio.run(run_answer_accuracy(sample))
    return answer_accuracy_score


if __name__ == '__main__':
    response = "The Eiffel Tower is located in Paris."
    reference = "The Eiffel Tower is located in Paris. It has a height of 1000ft."

    semantic_similarity_score = compute_ragas_semantic_similarity(response, reference)
    print(f"Ragas SemanticSimilarity: {semantic_similarity_score}")

    user_input = "When was Einstein born?"
    response = "Albert Einstein was born in 1879."
    reference = "Albert Einstein was born in 1879."

    answer_accuracy_score = compute_ragas_answer_accuracy(user_input, response, reference)
    print(f"Ragas AnswerAccuracy: {answer_accuracy_score}")


