# Cook-MultiPlayer

**COOK-MultiPlayer** is a collaborative game environment built upon **Overcooked-AI**, which extends the agent architecture of **[ProAgent](https://github.com/PKU-Alignment/ProAgent)** to support cooperative task performance.

## Overview

This project extends [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai), a benchmark environment for fully cooperative human-AI task performance based on the popular video game Overcooked. The framework enables research and experimentation with multi-agent collaboration strategies using LLM-powered agents.

## Project Structure

```
.
├── lib/
│   └── overcooked_ai/          # Overcooked-AI environment
│       ├── overcooked_ai_py/   # Python implementation
│       └── setup.py
├── src/
│   ├── main.py                 # Main entry point
│   ├── logger.py               # Logging utility
│   ├── exp.sh                  # Example experiment script
│   ├── proagent/               # ProAgent implementation
│   │   ├── proagent.py         # Main ProAgent class
│   │   ├── modules.py          # modules
│   │   └── utils.py            # Utility functions
│   ├── prompts/                # Prompt templates
│   │   ├── cook.csv            # Player profiles
│   │   └── refine_*.txt       # Prompts for players and agents
│   └── utils.py                # General utilities
└── README.md
```

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n proagent python=3.7
conda activate proagent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
conda install mpi4py==3.1.4
```

### Step 3: Install Overcooked-AI

```bash
cd ./lib/overcooked_ai
pip install -e .
cd ../..
```

## Usage

### API Configuration

When using ProAgent with custom LLM deployments, you need to configure API endpoints for both the agent (agent_index=0) and player (agent_index=1). Edit `src/exp.sh` and replace the placeholder values:

```bash
--agent_url "YOUR_AGENT_DEPLOYMENT_URL"      # API URL for agent (agent_index=0)
--agent_model "YOUR_AGENT_MODEL"            # Model name for agent
--agent_api_key "YOUR_AGENT_API_KEY"         # API key for agent
--player_url "YOUR_PLAYER_DEPLOYMENT_URL"    # API URL for player (agent_index=1)
--player_model "YOUR_PLAYER_MODEL"           # Model name for player
--player_api_key "YOUR_PLAYER_API_KEY"       # API key for player
```


### Using the Example Script

```bash
cd ./src
bash exp.sh
```

The `exp.sh` script runs:
```bash
python main.py \
  --layout asymmetric_advantages \
  --p0 ProAgent \
  --p1 ProAgent \
  --horizon 400 \
  --using_big_5 True \
  --level 0 \
  --agent_url "YOUR_AGENT_DEPLOYMENT_URL" \
  --agent_model "YOUR_AGENT_MODEL" \
  --agent_api_key "YOUR_AGENT_API_KEY" \
  --player_url "YOUR_PLAYER_DEPLOYMENT_URL" \
  --player_model "YOUR_PLAYER_MODEL" \
  --player_api_key "YOUR_PLAYER_API_KEY"
```