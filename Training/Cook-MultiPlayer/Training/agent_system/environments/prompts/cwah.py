CWAH_ACTION_TEMPLATE ="""
I'm {AGENT_NAME}. I'm in a hurry to finish the housework with my friend {OPPO_NAME} together. Given our shared goal, dialogue history, and my progress and previous actions, please help me choose the best available action to achieve the goal as soon as possible. Note that I can hold two objects at a time and there are no costs for holding objects. All objects are denoted as <name> (id), such as <table> (712).
Goal: {GOAL}
Progress: {PROGRESS}
Dialogue history:
Alice: ""Hi, I'll let you know if I find any goal objects and finish any subgoals, and ask for your help when necessary.""
Bob: ""Thanks! I'll let you know if I find any goal objects and finish any subgoals, and ask for your help when necessary.""
{DIALOGUE_HISTORY}
Previous actions: {ACTION_HISTORY}
Available actions:
{AVAILABLE_ACTIONS}

Now it's your turn to take an action.
1. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. The resoning process should be helpful and brief. 200 words at most.
2. Once you've finished your reasoning, you must also generate a short message to send to {OPPO_NAME} to help us achieve the goal as soon as possible, enclosed within <message> </message> tags. The generated message should be accurate, helpful and brief. Do not generate repetitive messages. If you choose <send_message> in the following action selection, the content of this message will be sent to {OPPO_NAME}.
3. Once you've finished your message generation, you should choose exactly an admissible action in available actions above for current step and present it within <action> </action> tags.

Example format:
<think>Your resoning content</think>
<message>Your generated message</message>
<action>Your selected action</action>
"""

CWAH_CHAT_TEMPLATE = """
I'm {AGENT_NAME}. I'm in a hurry to finish the housework with my friend {OPPO_NAME} together. Given our shared goal, dialogue history, and my progress and previous actions, please help me generate a short message to send to {OPPO_NAME} to help us achieve the goal as soon as possible. Note that I can hold two objects at a time and there are no costs for holding objects. All objects are denoted as <name> (id), such as <table> (712).
Goal: {GOAL}
Progress: {PROGRESS}
Previous actions: {ACTION_HISTORY}
Dialogue history:
Alice: ""Hi, I'll let you know if I find any goal objects and finish any subgoals, and ask for your help when necessary.""
Bob: ""Thanks! I'll let you know if I find any goal objects and finish any subgoals, and ask for your help when necessary.""
{DIALOGUE_HISTORY}

Note: The generated message should be accurate, helpful and brief. Do not generate repetitive messages.
"""