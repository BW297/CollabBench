# CWAH-MultiPlayer Training

This document describes how to train agents for **CWAH-MultiPlayer** (Communicative Watch-And-Help), a collaborative benchmark built on VirtualHome for multi-agent communication and coordination. Training is built on the [**verl-agent**](https://github.com/langfengQ/verl-agent) framework, which is an extension of [veRL](https://github.com/volcengine/verl) for training LLM/VLM agents via reinforcement learning (RL).

## Overview

- **verl-agent** provides step-independent multi-turn rollouts, customizable memory, and RL algorithms such as **GiGPO** (Group-in-Group Policy Optimization).
- CWAH-MultiPlayer extends the Watch-And-Help challenge with explicit communication between agents and is built on **VirtualHome**.
- Training uses GiGPO with the CWAH environment.

## Installation

Install in two stages: first the verl-agent base and dependencies, then the CWAH-MultiPlayer environment.

### Step 1: verl-agent base installation

Follow the installation and configuration in the Training README (verl-agent codebase and environment setup):

- **[VeRL](Training/README.md)**

Complete the “Install veRL” section and any environment-specific steps you need before proceeding.

### Step 2: CWAH-MultiPlayer (VirtualHome) setup

Then install and configure the CWAH-MultiPlayer running environment (VirtualHome API and simulator):

- **[CWAH-MultiPlayer (Running) README](../../Evaluation/Running/CWAH-MultiPlayer/README.md)** — VirtualHome clone, executable download, conda env, and dependencies.

First go into `Training/agent_system/environments/env_package/cwah`, then install and configure according to the CWAH-MultiPlayer (Running) README.

```bash
cd Training/agent_system/environments/env_package/cwah
# Then install and configure according to the CWAH-MultiPlayer (Running) README
```
Ensure the final directory layout includes the game engine files and executables:

```bash
├── Training/agent_system/environments/env_package/cwah/
│   ├── cwah/                 
│   ├── executable/ 
│   ├── virtualhome/                
│   └── ...  
```

Ensure the CWAH backend (API server used by environment workers) is available at the URL you will set in the training script (see below).

## Running the Training Script

1. Go into `Training`:

```bash
cd Training
```

2. Start GiGPO training with the CWAH script:

Before running the training script, you need to configure the following parameters in `examples/gigpo_trainer/run_cwah.sh`:

- **`LOG_DIR`**: Directory path where training logs will be saved.

- **`TRAIN_MODEL_PATH`**: Path to the base LLM model used for training (actor model).

- **`CWAH_URL`**: Base URL for the LLM used by the ProAgent environment (p1 agent) during training, and by interactivity evaluation.

- **`CWAH_MODLE_ID`**: Model name for the LLM used by ProAgent (p1 agent) during training, and by interactivity evaluation.

```bash
bash examples/gigpo_trainer/run_cwah.sh
```
