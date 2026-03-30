>W4

SFT results of CollabBench in CWAH in both CB-Efficiency and CB-Affective ($P_{target}$ serves as Agent 1).

| Label | Efficiency Score | Efficiency Std | Tokens K | Helpfulness Mean | Trustfulness Mean | Empathy Mean |
| ----- | ---------------- | -------------- | -------- | ---------------- | ----------------- | ------------ |
| BASE  | 84.51            | 33.23          | 0.24     | 1.22             | 2.58              | 2.50         |
| SFT   | 80.31            | 32.74          | 0.25     | 1.31             | 2.77              | 2.53         |
| RL    | **71.64**        | **25.16**      | **0.23** | **1.43**         | **3.03**          | **3.33**     |

> Q1

Spearman rank correlation between model scale and human rating consistency in CWAH, evaluated across five representative player types across five tasks (0, 10, 20, 30, 40).

| Model Scale | Personality Filtering | Collaborative Evaluation |              |          |
| ----------- | --------------------- | ------------------------ | ------------ | -------- |
|             | Consistency           | Helpfulness              | Trustfulness | Empathy  |
| Qwen2.5-14B | 0.54                  | 0.28                     | 0.43         | 0.74     |
| Qwen2.5-32B | 0.70                  | 0.40                     | 0.56         | 0.74     |
| Qwen2.5-72B | **0.76**              | **0.63**                 | **0.63**     | **0.78** |

Spearman rank correlation between model type and human rating consistency in CWAH, evaluated across five representative player types across five tasks (0, 10, 20, 30, 40).

| Model Type  | Personality Filtering | Collaborative Evaluation |              |          |
| ----------- | --------------------- | ------------------------ | ------------ | -------- |
|             | Consistency           | Helpfulness              | Trustfulness | Empathy  |
| GPT-5.2     | **0.81**              | 0.65                     | **0.83**     | **0.87** |
| DeepSeek    | 0.79                  | **0.75**                 | 0.77         | 0.74     |
| Qwen2.5-72B | 0.76                  | 0.63                     | 0.63         | 0.78     |

> Q4

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

>Q6

Diversity evaluation based on trajectory encoding by BGE-M3 in CWAH.

| CWAH                               | BGE-M3 | Qwen3-Embedding-4B |
| ---------------------------------- | ------ | ------------------ |
| Cluster (↑) | 11.4   | 17.6               |
| Spread (↑)                | 0.62   | 0.74               |

> L1

Evaluation results on long-horizon collaborative writing and programming tasks in CollabBench (30 conversations evaluated, Qwen2.5-7B-Instruct).


| Tasks | Metrics | Base | CollabBench |
| :-- | --- | --- | --- |
|  MediumDocEdit-Chat   | Document->Bleu（↑） | **0.52** | 0.49 |
|   | Interactivity（↑） | 0.78 | **0.89** |
|  | Token Amount ($k$) （↓） | 2.75 | **2.56** |
|  MATH-Chat | ACC（↑） | **0.93** | 0.91 |
|  | Interactivity（↑） | 0.61 | **0.68** |
|  | Token Amount ($k$) （↓） | 1.58 | **1.51** |

