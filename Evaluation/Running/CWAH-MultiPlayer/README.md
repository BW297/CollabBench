# CWAH-MultiPlayer

**CWAH-MultiPlayer** is a collaborative benchmark based on **Communicative Watch-And-Help (C-WAH)**, built on top of **VirtualHome** to study multi-agent communication and coordination.

## Overview

C-WAH extends the Watch-And-Help challenge with explicit communication between agents. Messages are treated as actions (one timestep each) with a fixed length budget, enabling research on coordination strategies in household task planning.

## Project Structure

```
.
├── LLM/                        # LLM agent logic and prompts
├── MCTS/                       # MCTS baseline implementation
├── agents/                     # Agent wrappers and interfaces
├── algos/                      # Planning/learning algorithms
├── dataprocess/                # Data processing scripts
├── dataset/                    # Task data and assets
├── envs/                       # Environment wrappers
├── gen_data/                   # Data generation utilities
├── scripts/                    # Experiment scripts
├── testing_agents/             # Agent evaluation scripts
├── utils/                      # Shared utilities
├── virtualhome_userinterface/  # Human-facing UI
├── arguments.py                # Argument parser
├── requirements.txt
└── README.md
```

## Installation

### Step 1: Get the VirtualHome Simulator and API

Clone the **VirtualHome API** (use the `wah` branch), and download the Linux executable.

```bash
git clone --branch wah https://github.com/xavierpuigf/virtualhome.git
```

```bash
gdown https://drive.google.com/uc?id=1L79SxE07Jt-8-_uCvNnkwz5Kf6AjtaGp
unzip executable.zip
chmod +x executable/linux_exec.v2.3.0.x86_64
```

Recommended directory layout:

```bash
|--CWAH-MultiPlayer/
|--virtualhome/
|--executable/
```

### Step 2: Install Requirements

```bash
cd CWAH-MultiPlayer
conda create --name cwah python=3.8
conda activate cwah
pip install -r requirements.txt
```

## Usage

Example experiment scripts live in `scripts/`. For two LLM agents:

```bash
./scripts/symbolic_obs_llm_llm.sh
```

See `arguments.py` and the scripts for full argument options.

## Evaluation

The evaluation pipeline lives at the repository root: `../evaluation/`.

## User Interface

To launch the UI for human experiments:

```bash
cd virtualhome_userinterface
python vh_demo.py --deployment remote --executable_file ../../executable/linux_exec.v2.3.0.x86_64 --extra_agent MCTS_comm --task_group 0 --showmodal
```

## Environment Details

Five C-WAH task families are supported: `Prepare afternoon tea`, `Wash dishes`, `Prepare a meal`, `Put groceries`, and `Set up a dinner table`. Each task is defined by a set of `ON/IN(x, y)` predicates, and the goal is to satisfy all predicates within 250 steps.

### Metrics

- **Average Steps (L)**: steps to finish the task
- **Efficiency Improvement (EI)**: gains from cooperating with base agents
