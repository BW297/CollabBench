> W2

3. Standard deviation of other metrics for CollabBench across different models.

Standard deviation of other metrics on CollabBench across different models in CWAH.

| Model                       | Agent   | Helpfulness (std) | Trustfulness (std) | Empathy (std) | #Tokens (k, std) |
| --------------------------- | ------- | ----------------- | ------------------ | ------------- | ---------------- |
| GPT-5.2 Oracle              | Agent 1 | 0.869             | 0.420              | 0.384         | 0.014            |
| GPT-5.2 Oracle              | Agent 2 | 0.858             | 0.378              | 0.435         | 0.014            |
| GPT-5.2 Base                | Agent 1 | 1.21              | 0.45               | 0.33          | 0.03             |
| GPT-5.2 Base                | Agent 2 | 1.25              | 0.48               | 0.29          | 0.03             |
| DeepSeek-V3.1 Base          | Agent 1 | 0.752             | 0.394              | 0.361         | 0.020            |
| DeepSeek-V3.1 Base          | Agent 2 | 0.873             | 0.391              | 0.381         | 0.023            |
| Qwen2.5-72B-Instruct Base   | Agent 1 | 0.885             | 0.433              | 0.386         | 0.043            |
| Qwen2.5-72B-Instruct Base   | Agent 2 | 0.927             | 0.427              | 0.391         | 0.043            |
| Qwen2.5-7B-Instruct Base    | Agent 1 | 0.842             | 0.452              | 0.414         | 0.017            |
| Qwen2.5-7B-Instruct Base    | Agent 2 | 0.964             | 0.482              | 0.446         | 0.018            |
| Qwen2.5-7B-Instruct Trained | Agent 1 | 0.762             | 0.415              | 0.402         | 0.017            |
| Qwen2.5-7B-Instruct Trained | Agent 2 | 0.879             | 0.448              | 0.419         | 0.020            |

Standard deviation of other metrics on CollabBench across different models in COOK.

| Model                       | Agent   | Helpfulness (std) | Trustfulness (std) | Empathy (std) | #Tokens (k, std) |
| --------------------------- | ------- | ----------------- | ------------------ | ------------- | ---------------- |
| GPT-5.2 Oracle              | Agent 1 | 1.23              | 0.49               | 0.37          | 0.04             |
| GPT-5.2 Oracle              | Agent 2 | 1.31              | 0.46               | 0.34          | 0.03             |
| GPT-5.2 Base                | Agent 1 | 1.21              | 0.45               | 0.33          | 0.03             |
| GPT-5.2 Base                | Agent 2 | 1.25              | 0.48               | 0.29          | 0.03             |
| DeepSeek-V3.1 Base          | Agent 1 | 1.08              | 0.44               | 0.32          | 0.04             |
| DeepSeek-V3.1 Base          | Agent 2 | 1.12              | 0.46               | 0.33          | 0.04             |
| Qwen2.5-72B-Instruct Base   | Agent 1 | 0.85              | 0.38               | 0.27          | 0.03             |
| Qwen2.5-72B-Instruct Base   | Agent 2 | 0.90              | 0.40               | 0.29          | 0.03             |
| Qwen2.5-7B-Instruct Base    | Agent 1 | 0.55              | 0.33               | 0.23          | 0.02             |
| Qwen2.5-7B-Instruct Base    | Agent 2 | 0.57              | 0.35               | 0.25          | 0.02             |
| Qwen2.5-7B-Instruct Trained | Agent 1 | 0.58              | 0.35               | 0.26          | 0.02             |
| Qwen2.5-7B-Instruct Trained | Agent 2 | 0.62              | 0.36               | 0.28          | 0.02             |

> W3

Diversity evaluation in CWAH (Task 0: *read_book*), comparing CollabBench with multiple baseline models.

| CWAH | Multi LLM | CollabBench |
| --- | --- | --- |
| $ \text{Cluster}_ \xi $（$\uparrow$） | 32 | **36** |
| Spread($\uparrow$) | 0.78 | **0.97** |

