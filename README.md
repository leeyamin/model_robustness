# Auditing LLMs Robustness

## TL;DR
This repo audits how consistently an LLM answers the same question when it’s phrased in different ways. It generates probe questions, collects the model’s answers, and scores both consistency (robustness) and correctness (accuracy). It’s domain-agnostic by design; the included PoC showcases a medical QA dataset.

- **What you get**: Robustness and accuracy scores per question and per probe set.
- **How it works**: Rephrase the original question into multiple semantically similar probes, get answers from the audited model, then measure answer similarity and accuracy with Ragas-based metrics.

For quick setup and running instructions, see [howto.md](howto.md).


## Introduction

As Large Language Models (LLMs) continue to integrate deeply into our daily lives, 
ensuring their **robustness**, **reliability**, and **consistency** is crucial.
This project provides a comprehensive framework for auditing the robustness of LLMs. 
This framework is built upon the foundational work presented in ["AuditLLM: A Tool for Auditing Large Language Models Using Multiprobe Approach"](https://dl.acm.org/doi/abs/10.1145/3627673.3679222) by Amirizaniani et al. (2024), 
and extends it with essential novel enhancements to improve robustness and reliability.

## Motivation

Current LLM tests often use rigid benchmarks, which might not fully capture how well models truly perform when users interact with them in diverse, real-world scenarios.
Users interact with models using a wide variety of questions and phrasing. 
A robust model must provide consistent and accurate answers even when a question is asked in slightly different ways.
For example, asking *"What are the side effects of penicillin?"* should ideally yield a consistent answer to *"Can penicillin cause any unwanted side effects?"*. 
Inconsistencies in these scenarios can undermine user trust and lead to unreliable outcomes in critical applications.
This project aims to close that gap.

## Overview

The proposed auditing framework is engineered for **domain-agnostic robustness evaluation**.
It is designed to assess LLM consistency across diverse question-and-answer pairs, independent of their domain or topic.
The framework employs multiple probe LLMs and similarity metrics to provide a comprehensive assessment of model robustness.
It is designed for versatile domain applicability, with this implementation showcasing its utility within the medical domain, as a focused
proof of concept.
This choice underscores the critical need for robustness in healthcare, where the implications of inconsistent or inaccurate LLM outputs could be extremely severe.
Nevertheless, the framework's modular design facilitates its use in any other domain, by providing domain-specific datasets.


## Technical Implementation

Building upon the foundational principles of the AuditLLM framework, this implementation introduces several key enhancements,
significantly advancing both robustness evaluation and accuracy assessment. The system's architecture comprises three primary components:


### Probe Generation System
This component is responsible for generating diverse yet semantically equivalent variations for a given set of input questions, 
enabling a domain-level robustness evaluation. 
This approach eliminates bias toward a single question, providing a more comprehensive assessment of an audited model's behavior across a specific domain. 
Building upon the methodology for probe generation, our system further extends the foundational approach by employing a multi-model strategy 
rather than relying on a single probing model. 
This enhancement addresses the potential for bias and provides a more robust and unbiased evaluation of the model under audit. 
The framework is agnostic to the specific probing LLMs used, allowing for the selection of multiple suitable language models 
to collaboratively create these variations while strictly preserving the core meaning of the original question. 
Within this enhanced framework, each generated probe undergoes rigorous validation via semantic similarity checks to ensure fidelity to the original intent
[PoC note: filtering is disabled due to local model constraints].
Additionally, to maximize coverage and prevent redundancy, this system applies clustering techniques to select the most diverse and representative set of probes. 
This comprehensive approach is critical for effectively uncovering subtle inconsistencies in the audited model's answers across different but semantically equivalent phrasings.

### Answer Analysis Engine
The Answer Analysis Engine systematically processes and evaluates the answers provided by the audited model. 
The system processes these answers through a series of rigorous quality checks. 
Initially, it filters out uninformative or generic responses (e.g., "I don't know," "I can't help with that") using a predefined exclusion list 
[PoC note: filtering is disabled due to local model constraints].
This is a key enhancement to ensure meaningful robustness scores: by removing such trivial answers, the framework prevents 
the audited model from achieving high consistency based on consistently irrelevant outputs.

To enhance operational resilience and data integrity, this component incorporates a robust retry mechanism. 
This mechanism automatically re-attempts response generation up to three times when the audited model encounters failures or returns empty outputs. 
This comprehensive approach ensures that our robustness assessment is founded exclusively on meaningful and unique answers from the audited model.

### Robustness Assessment Framework
This final component serves as the primary stage for evaluating the audited model's performance, synthesizing the processed answers into actionable insights. 
It employs a comprehensive metric system to thoroughly evaluate the model's consistency as well as its accuracy—a crucial extension to the AuditLLM methodology.

To achieve this, the framework calculates various similarity scores between answers. 
It generates diverse metrics, including those that quantify answer consistency across probes and direct accuracy comparisons against ground truth answers. 
These provide a nuanced understanding of model behavior across diverse inputs.

## Real-World Example

To illustrate our framework's capabilities, this section presents an example from a medical domain evaluation, 
showcasing its power in assessing an audited model's robustness.


| Component               | Content/Result                                                                                                                  | Explanation                   |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Original Question       | "What are the side effects of penicillin?"                                                                                      | Input question to evaluate    |
| Ground Truth            | "Common side effects of penicillin include nausea, diarrhea, and allergic reactions such as rashes."                            | Reference answer for accuracy |
| Model's Original Answer | "Penicillin can cause side effects like nausea, vomiting, and diarrhea. In some cases, it can also lead to allergic reactions." | Accuracy Score: 0.93          |


### Initial Performance & Probe Generation
The audited model's answer to the original question achieved an Accuracy Score of 0.93 against the ground truth. This high initial accuracy provides a baseline. 
To comprehensively test its robustness, our framework then employed its multi-model Probe Generation System to create 
diverse sets of semantically equivalent questions.

### Multi-Probe Evaluation & Answer Analysis

The audited model's answers to these generated probes undergo our Answer Analysis Engine's rigorous quality checks and filtering. 
We collect its responses to each probe, and then analyze both their consistency amongst themselves and their accuracy against the ground truth.

#### Probe Set 1 (Generated by Mistral-7B)
| Probe Question                                            | Audited Model Answer                                                                                                                                                        | Accuracy vs GT |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| "What are the potential adverse reactions to penicillin?" | "Potential adverse reactions include gastrointestinal issues such as nausea and diarrhea, as well as allergic responses ranging from skin rashes to more severe reactions." | 0.92           |
| "Can penicillin cause any unwanted side effects?"         | "Yes, penicillin can cause unwanted side effects including gastrointestinal problems like diarrhea and nausea. Allergic reactions such as rashes are also common."          | 0.90           |
|                                                           | **Answer Similarity Score: 0.89** (robustness between all answers in set)                                                                                                   |                |
**Set Robustness**: 89% (average semantic similarity between answers in this set)

#### Probe Set 2 (Generated by Zephyr-7B)
| Probe Question                                                | Audited Model Answer                                                                                                                                                    | Accuracy vs GT |
|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| "What negative reactions might occur from taking penicillin?" | "Common negative reactions to penicillin include digestive issues like nausea and diarrhea. Some patients may experience allergic reactions, particularly skin rashes." | 0.91           |
| "How does the body react adversely to penicillin?"            | "The body may react adversely to penicillin through digestive symptoms like nausea and diarrhea, or through allergic responses including skin rashes."                  | 0.89           |
|                                                               | **Answer Similarity Score: 0.91** (robustness between all answers in set)                                                                                               |                |
**Set Robustness**: 91% (average semantic similarity between answers in this set)

### Results Summary
Our Robustness Assessment Framework synthesizes these results, showing:
  - Overall Robustness Score: 90% (calculated as the average consistency across all probe sets). This indicates the audited model's high ability to provide consistently similar answers despite variations in question phrasing.
  - High Answer Quality: All individual answers, across both original and probe questions, maintained high accuracy against the ground truth (above 0.89). This verifies that the consistency achieved is not based on uninformative or incorrect responses.


This example clearly demonstrates how our framework:
  - Evaluates model performance comprehensively using a multi-probe, multi-LLM generation strategy.
  - Measures both consistency across varied inputs and accuracy against ground truth, ensuring reliable and meaningful robustness assessments.
  - Provides granular insights into an audited model's behavior under real-world questioning paradigms.

## Conclusion

This framework provides a robust and comprehensive solution for auditing Large Language Models, 
directly addressing the critical need for reliable and consistent AI in real-world applications. 
By integrating multi-model probe generation and rigorous answer analysis with both consistency and accuracy assessments, 
it offers a nuanced understanding of model behavior far beyond traditional evaluation methods. 
This allows developers and researchers to build and deploy LLMs with greater confidence, ensuring they perform reliably 
and correctly even under varied and challenging conditions.