<div align="center">

<h1>CollabBench: Benchmarking and Unleashing Collaborative Ability of LLMs with Diverse Players via Proactive Engagement</h1>

<p><strong>ICML 2026</strong></p>

<p>
  Hong Qian,
  Yuanhao Liu,
  Zihan Zhou,
  Zongbao Zhang,
  Hanjie Ge,
  Haotian Shi,
  Liang Dou,
  Xiangfeng Wang,
  Jingwen Yang*,
  and Aimin Zhou
</p>

<p>
  East China Normal University<br>
  Tencent Inc.<br>
  Shanghai Innovation Institute
</p>

<p>
  <a href="paper/CollabBench.pdf"><img src="https://img.shields.io/badge/Paper-PDF-orange" alt="Paper PDF"></a>
  <a href="https://github.com/BW297/CollabBench"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub Repository"></a>
  <a href="paper/CollabBench.pdf"><img src="https://img.shields.io/badge/Conference-ICML%202026-blue" alt="ICML 2026"></a>
  <a href="https://github.com/BW297/CollabBench"><img src="https://img.shields.io/badge/Focus-Collaborative%20LLM%20Agents-green" alt="Collaborative LLM Agents"></a>
</p>

<img src="image/framework.png" width="820" alt="CollabBench framework" />

</div>

------
## Overview

We propose **CollabBench**, a benchmark for **systematically evaluating and training LLM-based agents to proactively collaborate with diverse players**.

CollabBench focuses on collaborative agent research, aiming to facilitate research on LLM-based agents in **efficient and affective interactions**.

This repository is organized into the following **four sections**.

## Table of Contents

- [Diverse Player Profiles Simulation](#1️⃣-diverse-player-profiles-simulation)
- [Collaborative Agentic Training](#2️⃣-collaborative-agentic-training)
- [Evaluation](#3️⃣-evaluation)
- [Player Trajectory Demonstration](#4️⃣-player-trajectory-demonstration)
- [Citation](#-citation)



## 1️⃣ Diverse Player Profiles Simulation

```bash
cd Anthropomorphic
````

This section focuses on modeling **diverse player profiles** from trajectory data.

📄 **Details:** [Anthropomorphic](Anthropomorphic/README.md)

---

## 2️⃣ Collaborative Agentic Training

This section describe the **training of the collaborative agents** for the two multi-player game environments.

```bash
cd Training
````

### 🎮 CWAH-MultiPlayer

```bash
cd CWAH-MultiPlayer
```

📄 **Details:** [CWAH-MultiPlayer](Training/CWAH-MultiPlayer/README.md)


### 🎮 Cook-MultiPlayer

```bash
cd Cook-MultiPlayer
```

📄 **Details:** [Cook-MultiPlayer](Training/Cook-MultiPlayer/README.md)

---

## 3️⃣ Evaluation

This section describes the **trajectory data collection and affective LLM judge** used in CollabBench for the two multi-player game environments.

```bash
cd Evaluation
````

### trajectory data collection

```bash
cd Running
````

#### 🎮 CWAH-MultiPlayer

```bash
cd CWAH-MultiPlayer
```

📄 **Details:** [CWAH-MultiPlayer](Evaluation/Running/CWAH-MultiPlayer/README.md)


#### 🎮 Cook-MultiPlayer

```bash
cd Cook-MultiPlayer
```

📄 **Details:** [Cook-MultiPlayer](Evaluation/Running/Cook-MultiPlayer/README.md)

---

### Affective LLM Judge

```bash
cd Judge
```

📄 **Details:** [Evaluation](Evaluation/Judge/README.md)

---

## 4️⃣ Player Trajectory Demonstration

We visualize representative trajectories for **five typical player types** (GIF format) to illustrate their collaboration behaviors.

### ❶ Efficient Collaboration Expert

![gif-0](figure/0/gif-0.gif "Efficient Collaboration Expert")


### ❷ Hesitant Laggard

![gif-1](figure/1/gif-1.gif "Hesitant Laggard")


### ❸ Anxious Doubter

![gif-4](figure/4/gif-4.gif "Anxious Doubter")


### ❹ Proactive Leader

![gif-7](figure/7/gif-7.gif "Proactive Leader")


### ❺ Independent Loner

![gif-13](figure/13/gif-13.gif "Independent Loner")


## 💭 Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{CollabBench2026,
  author = {Hong Qian and Yuanhao Liu and Zihan Zhou and Zongbao Zhang and Hanjie Ge and Haotian Shi and Liang Dou and Xiangfeng Wang and Jingwen Yang and Aimin Zhou},
  title = {CollabBench: Benchmarking and Unleashing the Collaborative Ability of LLMs with Diverse Players via Proactive Engagement},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026},
  address = {Seoul, South Korea}
}
```

Reference:

> Hong Qian, Yuanhao Liu, Zihan Zhou, Zongbao Zhang, Hanjie Ge, Haotian Shi, Liang Dou, Xiangfeng Wang, Jingwen Yang, and Aimin Zhou. CollabBench: Benchmarking and Unleashing the Collaborative Ability of LLMs with Diverse Players via Proactive Engagement. In Proceedings of the 43rd International Conference on Machine Learning, 2026.