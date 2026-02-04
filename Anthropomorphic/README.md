# Player Profile Summarization

This module summarizes player behavioral profiles from trajectory data using LLM-based analysis. It processes player trajectories, divides them into time windows, and generates concise behavioral pattern descriptions.

## Overview

The summarization process:
1. Reads trajectory data from clustered sessions
2. Divides trajectories into three-step time windows (Example IDs)
3. Uses LLM prompts to identify common behavioral patterns
4. Generates concise profile descriptions for each cluster

## Project Structure

```
.
├── summarize.py          # Main script for profile summarization
├── Cook_prompt.txt       # Prompt template for profile summarization
├── data/                 # Trajectory data directory
│   ├── Cluster1/         # Cluster 1
│   │   └── actions.json # Trajectory steps for this cluster
│   ├── Cluster2/         # Cluster 2
│   │   └── actions.json
│   └── ...
└── output.json          # Generated profile summaries
```

## Data Format

### Directory Structure

Each subdirectory in `data/` represents a **cluster** containing aggregated trajectory steps from multiple game sessions. The folder name identifies the cluster.

### actions.json Format

Each `actions.json` file is a list of trajectory steps:

```json
[
    {
        "think": "...",
        "action": "...",
        "message": "...",
        ...
    },
    {
        "think": "...",
        "action": "...",
        "message": "...",
        ...
    },
    ...
]
```


## How to Build Data in the `data/` Folder

### Step 1: Organize Clusters

Create a subdirectory for each cluster you want to analyze:

```bash
mkdir -p data/Cluster1
mkdir -p data/Cluster2
mkdir -p data/Cluster3
# ... etc
```

### Step 2: Aggregate Trajectory Steps

For each cluster, collect trajectory steps from multiple game sessions and aggregate them into a single `actions.json` file.

**Important**: 
- Each folder represents a **cluster** containing aggregated trajectory steps
- All trajectory steps from games in the same cluster should be combined
- The steps should be ordered chronologically (by timestep)

### Step 3: Structure the JSON File

Create `actions.json` in each cluster folder as a **list** of trajectory steps:

```json
[
    {
        "think": "...",
        "action": "...",
        "message": "...",
    },
    ...
]
```

### Step 4: Run Summarization

```bash
python summarize.py
```

## Prompt Template

The prompt template is stored in `Cook_prompt.txt` and `CWAH_prompt.txt`. You can choose the appropriate prompt based on the game environment. The prompt uses `$trajectory_steps` as a placeholder that will be replaced with the formatted trajectory data during execution.
