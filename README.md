# Large Language Models Robustness Auditing Framework

This project provides a framework for auditing the robustness of Large Language Models (LLMs). 
As these models become more integrated into our daily lives, it's crucial to ensure they are not only accurate but also reliable and consistent. 
This framework is inspired by the methodology introduced in the paper ["AuditLLM: A Tool for Auditing Large Language Models Using Multiprobe Approach"](https://dl.acm.org/doi/10.1145/3688688.3688972) by Amirizaniani et al. (2024) 
and extends it with several enhancements to improve robustness and reliability.

The auditing framework itself is domain-agnostic. 
The methodology can be applied to evaluate model robustness on any set of questions and answers, 
regardless of the subject matter.

The choice of the medical domain for our POC was deliberate. 
In fields like healthcare, the cost of inconsistent or inaccurate information from an LLM is particularly high, making robustness a critical requirement. 
The framework can be easily extended to other domains such as finance, law, or education by simply providing a different dataset. 
Furthermore, robustness scores can be aggregated across multiple domains to provide a holistic and comprehensive understanding of a model's overall reliability.

## Motivation

Traditional LLM evaluation often relies on static benchmarks, which may not fully capture how a model behaves in the real world. 
Users interact with models using a wide variety of questions and phrasing. 
A robust model should provide consistent and accurate answers even when a question is asked in slightly different ways.
For example, asking *"What are the side effects of penicillin?"* should yield a similar answer to *"Can penicillin cause any unwanted side effects?"*. 
Inconsistencies in these scenarios can erode user trust and lead to unreliable outcomes in critical applications. This framework is designed to systematically measure and quantify this aspect of model behavior.

## Auditing Methodology

The evaluation process is designed to test a model's consistency and accuracy through a multi-probe approach:

1.  **Start with a Core Question**: The audit begins with a set of well-defined questions and their corresponding "ground-truth" answers for a specific knowledge domain.
2.  **Generate Probes**: For each core question, we use other LLMs to generate several semantically similar but syntactically different versions. This creates a diverse set of "probes" that test the same underlying concept.
3.  **Challenge the Model**: The model being audited is then tasked with answering both the original question and all the generated probes.
4.  **Measure Consistency (Robustness Score)**: The answers to the probes are semantically compared with one another. A high degree of similarity among the answers results in a high **Robustness Score**, indicating the model is consistent.
5.  **Measure Correctness (Accuracy Score)**: Each answer is also compared to the original ground-truth answer to measure its factual accuracy, resulting in an **Accuracy Score**.
6.  **Generate a Detailed Report**: The complete findings, including all questions, answers, and scores, are compiled into a comprehensive report, allowing for a deep analysis of the model's behavior.

## Audit Report Example

The final output provides a detailed breakdown of the evaluation. 
Below is a simplified example of what a few rows from the report might look like for a single original question:

<table>
  <thead>
    <tr>
      <th>Original Question</th>
      <th>GT Answer</th>
      <th>Model Answer (MA)</th>
      <th>MA Accuracy</th>
      <th>Probe LLM</th>
      <th>Probe Question</th>
      <th>Probe Answer</th>
      <th>Probe Accuracy</th>
      <th>Robustness Score</th>
      <th>Avg Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" style="vertical-align: middle;">What are the side effects of penicillin?</td>
      <td rowspan="3" style="vertical-align: middle;">Common side effects of penicillin include nausea, diarrhea, and allergic reactions such as rashes.</td>
      <td rowspan="3" style="vertical-align: middle;">Penicillin can cause side effects like nausea, vomiting, and diarrhea. In some cases, it can also lead to allergic reactions.</td>
      <td rowspan="3" style="vertical-align: middle;">0.93</td>
      <td rowspan="2" style="vertical-align: middle;">TheBloke/Llama-2-7B-Chat-GGML</td>
      <td>What are the potential adverse reactions to penicillin?</td>
      <td>Potential adverse reactions to penicillin include gastrointestinal issues such as nausea and diarrhea, as well as allergic responses ranging from skin rashes to more severe anaphylactic shock.</td>
      <td>0.92</td>
      <td rowspan="2" style="vertical-align: middle;">89</td>
      <td rowspan="2" style="vertical-align: middle;">0.91</td>
    </tr>
    <tr>
      <td>Can penicillin cause any unwanted side effects?</td>
      <td>Yes, penicillin can cause unwanted side effects. The most common are gastrointestinal problems like diarrhea and nausea. It's also possible to have an allergic reaction, such as a rash or hives.</td>
      <td>0.90</td>
    </tr>
    <tr>
      <td>TheBloke/Mistral-7B-Instruct-v0.1-GGUF</td>
      <td>What negative reactions can occur from taking penicillin?</td>
      <td>Negative reactions from penicillin can include digestive upset, such as nausea or diarrhea. Allergic reactions, like skin rashes, are also a known possibility for some individuals.</td>
      <td>0.88</td>
      <td>91</td>
      <td>0.88</td>
    </tr>
  </tbody>
</table>

This structured output allows for as additional in-depth analysis of a model's behavior, helping to identify weaknesses in both its robustness and factual accuracy.

For more information, please refer to the design document: [Model Robustness Auditing Design Document](https://docs.google.com/document/d/1V7qDTZWVRgjS1_35j7pNnAuL-NCwDy8LnhlSbJwzf1U/edit?usp=sharing).