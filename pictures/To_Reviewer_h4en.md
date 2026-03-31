>Q1&W1

Evaluation results of small-scale open-source models on CollabBench in CWAH, assessed across five representative player types and five tasks (0, 10, 20, 30, 40) ($P_{target}$ serves as Agent 1).

| Series | Label                 | Efficiency Score | Efficiency Std | Tokens (k) | Helpfulness Mean | Trustfulness Mean | Empathy Mean |
| ------ | --------------------- | ---------------- | -------------- | ------------ | ---------------- | ----------------- | ------------ |
| Qwen   | Qwen2.5-3B            | 86.38            | 33.68          | 0.30         | 0.64             | 1.76              | 1.94         |
| Qwen   | Qwen2.5-7B-Instruct   | 84.51            | 33.23          | 0.24         | 1.22             | 2.58              | 2.50         |
| Qwen   | Qwen3-8B              | 80.43            | 30.25          | 2.99         | 0.62             | 2.09              | 1.91         |
| LLaMA  | Llama-3.2-3B-Instruct | 85.43            | 33.31          | 0.28         | 0.64             | 1.91              | 1.93         |
| LLaMA  | Llama-3.1-8B-Instruct | 83.19            | 32.40          | 0.40         | 0.82             | 2.23              | 2.14         |

> Q4

Example evaluation results in COOK forced_coordination scenario under prompt perturbations.

|        | synonyms / word order | behavior change  |
| ------ | --------------------- | ---------------- |
| Before | 120 （6 orders）      | 120 （6 orders） |
| After  | 120（6 orders）       | 80 （4 orders）  |

