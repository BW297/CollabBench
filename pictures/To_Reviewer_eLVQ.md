>W2&Q1

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

>W3&Q3

Evaluation results of small-scale open-source models on CollabBench in CWAH, assessed across five representative player types and five tasks (0, 10, 20, 30, 40) ($P_{target}$ serves as Agent 1).

| Series | Label                 | Efficiency Score | Efficiency Std | Tokens ($k$) | Helpfulness Mean | Trustfulness Mean | Empathy Mean |
| ------ | --------------------- | ---------------- | -------------- | ------------ | ---------------- | ----------------- | ------------ |
| Qwen   | Qwen2.5-3B            | 86.38            | 33.68          | 0.30         | 0.64             | 1.76              | 1.94         |
| Qwen   | Qwen2.5-7B-Instruct   | 84.51            | 33.23          | 0.24         | 1.22             | 2.58              | 2.50         |
| Qwen   | Qwen3-8B              | 80.43            | 30.25          | 2.99         | 0.62             | 2.09              | 1.91         |
| LLaMA  | Llama-3.2-3B-Instruct | 85.43            | 33.31          | 0.28         | 0.64             | 1.91              | 1.93         |
| LLaMA  | Llama-3.1-8B-Instruct | 83.19            | 32.40          | 0.40         | 0.82             | 2.23              | 2.14         |

>W4&Q2

+ Generalization between our two tasks: 

  Evaluation results in COOK using CollabBench trained on CWAH ($P_{target}$ serves as Agent 1).

  | Label   | Efficiency Score | Efficiency Std | Tokens Amount | Helpfulness | Trustfulness | Empathy  |
  | ------- | ---------------- | -------------- | ------------- | ----------- | ------------ | -------- |
  | BASE    | 86.93            | 35.30          | **0.23**      | 0.45        | 1.92         | 1.86     |
  | CWAH-RL | 93.60            | 41.68          | 0.30          | 0.67        | 2.15         | 1.97     |
  | RL      | **99.20**        | **34.03**      | **0.23**      | **0.74**    | **2.26**     | **2.12** |

+ Generalization in more realistic application beyond online games: 

Evaluation results on long-horizon collaborative writing and programming tasks in CollabBench (30 conversations evaluated).


| Tasks              | Metrics                               | Base     | CollabBench |
| :----------------- | ------------------------------------- | -------- | ----------- |
| MediumDocEdit-Chat | Document->Bleu（$ \uparrow $）        | **0.52** | 0.49        |
|                    | Interactivity（$ \uparrow $）         | 0.78     | **0.89**    |
|                    | Token Amount ($k$) （$ \downarrow $） | 2.75     | **2.56**    |
| MATH-Chat          | ACC（$ \uparrow $）                   | **0.93** | 0.91        |
|                    | Interactivity（$ \uparrow $）         | 0.61     | **0.68**    |
|                    | Token Amount ($k$) （$ \downarrow $） | 1.58     | **1.51**    |

