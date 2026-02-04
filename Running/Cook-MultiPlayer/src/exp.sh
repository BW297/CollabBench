#!/bin/bash
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