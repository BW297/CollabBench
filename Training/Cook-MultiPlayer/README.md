# Cook-MultiPlayer Training

This document describes how to train agents for **Cook-MultiPlayer** (a collaborative Overcooked-AI–based environment). Training is built on the [**verl-agent**](https://github.com/langfengQ/verl-agent) framework, which is an extension of [veRL](https://github.com/volcengine/verl) for training LLM/VLM agents via reinforcement learning (RL).

## Overview

- **verl-agent** provides step-independent multi-turn rollouts, customizable memory, and RL algorithms such as **GiGPO** (Group-in-Group Policy Optimization).
- Cook-MultiPlayer extends [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) and integrates with [ProAgent](https://github.com/PKU-Alignment/ProAgent) for cooperative multi-agent tasks.
- Training uses GiGPO with the ProAgent environment.

## Installation

Install in two stages: first the verl-agent base and dependencies, then the Cook-MultiPlayer (Overcooked-AI + ProAgent) environment.

### Step 1: verl-agent base installation

Follow the installation and configuration in the CWAH-MultiPlayer Training README (same verl-agent codebase and environment setup):

- **[VeRL](Training/README.md)**

Complete the “Install veRL” section and any environment-specific steps you need before proceeding.

### Step 2: Cook-MultiPlayer Overcooked-AI setup

Then install and configure the Cook-MultiPlayer running environment:

- **[Cook-MultiPlayer](../../Evaluation/Running/Cook-MultiPlayer/README.md)** — conda env, dependencies, Overcooked-AI, and ProAgent usage.

Ensure you can run the example script under “Usage” (e.g. `cd src && bash exp.sh`) before starting training.

## Running the Training Script

1. Go into `Training`:

```bash
cd Training
```

2. Start GiGPO training with the ProAgent script:

Before running the training script, you need to configure the following parameters in `examples/gigpo_trainer/run_proagent.sh`:

- **`LOG_DIR`**: Directory path where training logs will be saved.

- **`TRAIN_MODEL_PATH`**: Path to the base LLM model used for training (actor model).

- **`COOK_URL`**: Base URL for the LLM used by the ProAgent environment (p1 agent) during training, and by interactivity evaluation.

- **`COOK_MODLE_ID`**: Model name for the LLM used by ProAgent (p1 agent) during training, and by interactivity evaluation.

```bash
bash examples/gigpo_trainer/run_proagent.sh
```

