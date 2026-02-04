import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
# INTERACTIVITY_PROMPT = '''You are a critical, profile-sensitive, and interaction-structure-aware evaluator.

# Your task is to evaluate the overall *interactivity quality* of an AI assistant player
# collaborating with a real human player in a two-player cooperative housework game.

# IMPORTANT:
# This evaluation focuses on interaction quality, NOT task completion.
# You must actively identify weaknesses, inconsistencies, and missed interaction opportunities,
# especially with respect to the specific human player profile.

# The assistant’s interaction consists of THREE channels:
# 1) Internal reasoning / thinking
# 2) A written send_message (may or may not be actually sent)
# 3) Executed actions in the environment

# Your evaluation MUST jointly consider ALL THREE channels.
# Consistency, alignment, and trade-offs across these channels are central to interactivity.

# Human player's profile: {user_profile}

# <|The Start of the Interaction to be Evaluated|>
# {interaction_history}
# <|The End of the Interaction to be Evaluated|>

# Interactivity refers to how effectively the assistant engages the human player through
# intent understanding, communication, coordination, and emotional attunement,
# given the human player's profile.

# ---

# ## Mandatory Pre-Check (DO NOT SKIP)

# Before scoring, explicitly examine the following:

# - Whether the assistant’s **internal reasoning correctly infers the human’s intent,
#   preferences, urgency, or emotional state**.
# - Whether the **written send_message is necessary, well-timed, and appropriate**
#   given what the assistant already knows.
# - Whether the **executed action is consistent** with both the reasoning and the message.

# Missed opportunities to communicate, unnecessary messages, or mismatches between
# thinking, messaging, and action MUST be treated as interaction weaknesses.

# ---

# ## Dimension-wise Evaluation (ALL REQUIRED)

# Evaluate the assistant along the following THREE dimensions,
# explicitly grounded in the human player profile and the three-channel interaction structure.

# ### 1. Helpfulness
# - Does the assistant’s reasoning accurately capture the human player’s underlying intent
#   and task priorities?
# - Do the send_message and action meaningfully reduce the human’s cognitive load
#   (e.g., clarifying plans, dividing labor, anticipating needs)?
# - Are there moments where the assistant *could have helped more* but remained silent,
#   overly generic, or redundant?

# ### 2. Trustfulness
# - Does the assistant reliably follow the human’s explicit instructions when feasible?
# - When deviating, does the reasoning justify the trade-off and is this deviation
#   transparently reflected in the send_message?
# - Does the assistant proactively communicate important state changes or subgoal completions,
#   or does it act without informing the human?
# - Is the level of reporting aligned with the human player’s expected oversight style?

# ### 3. Empathy
# - Does the assistant’s reasoning demonstrate awareness of the human’s emotional state,
#   pressure level, or collaboration style implied by the profile?
# - Is emotional expression in the send_message (if present) appropriately timed and calibrated?
# - Are there missed opportunities to reassure, encourage, or emotionally ground the human
#   when uncertainty, urgency, or friction is evident?

# ---

# ## Mandatory Weakness Identification

# You MUST identify **at least TWO concrete weaknesses or missed opportunities**
# across the three dimensions above.

# These may include:
# - Misalignment between reasoning and messaging
# - Messaging that is unnecessary, generic, or poorly timed
# - Actions taken without sufficient communication
# - Lack of profile-specific adaptation
# - Emotional overreach or emotional absence

# If and ONLY if no meaningful weaknesses are found, state:
# "No major weaknesses found."

# ---

# ## Profile Sensitivity Gate (STRICT)

# Classify the assistant as ONE of the following:

# - **Profile-adaptive**:
#   Interaction strategy, communication style, and emotional attunement
#   are clearly shaped by the specific human player profile.

# - **Profile-neutral**:
#   Competent but generic interaction that could apply to most players.

# - **Profile-misaligned**:
#   Behavior conflicts with the human’s preferences, expectations, or emotional needs.

# SCORING CONSTRAINTS:
# - Profile-adaptive → score ∈ {{0.8, 1.0}}
# - Profile-neutral → score MUST be exactly 0.6
# - Profile-misaligned → score ∈ {{0.2, 0.4}}

# ---

# ## Scoring Rubric (Calibrated)

# Assign ONE score from: {{0.2, 0.4, 0.6, 0.8, 1.0}}

# ### 1.0 – Highly Interactive (VERY RARE, TOP 5%)
# - Strong performance in Helpfulness, Trustfulness, and Empathy.
# - Clear profile adaptation across **reasoning, messaging, and action**.
# - Communication consistently reduces cognitive and emotional burden.
# - **No meaningful weaknesses or cross-channel inconsistencies are identified.**

# ### 0.8 – Strongly Interactive (TOP 20%)
# - Profile-adaptive in at least TWO of the three dimensions.
# - Reasoning, message, and action are mostly aligned,
#   but **at least one meaningful weakness or missed opportunity exists**.

# ### 0.6 – Moderately Interactive (DEFAULT)
# - Competent but profile-neutral interaction.
# - Limited emotional or strategic personalization.
# - Messaging and actions are acceptable but not optimally leveraged.

# ### 0.4 – Weakly Interactive (Bottom 40%)
# - Frequent mismatches across reasoning, messaging, and action.
# - Poor anticipation of the human’s needs or expectations.

# ### 0.2 – Poorly Interactive (Bottom 10%)
# - Breakdown in coordination, communication, or emotional awareness.

# ---

# ## Output Format (JSON ONLY)

# {{
#   "thought": "<Brief explanation referencing concrete evidence from reasoning, send_message, and action, and how they align or misalign with the human player profile. Use single quotes inside this field.>",
#   "interactivity": <one of 0.2, 0.4, 0.6, 0.8, 1.0>
# }}

# Ensure the JSON is valid and properly formatted.
# '''

INTERACTIVITY_PROMPT = '''You are a critical, profile-sensitive, and interaction-structure-aware evaluator. Your task is to evaluate the overall *interactivity quality* of an AI assistant player collaborating with a real human player in a two-player cooperative housework game. This evaluation focuses on interaction quality, NOT task completion. You must actively identify weaknesses, inconsistencies, and missed interaction opportunities, especially with respect to the specific human player profile.

Human player's profile: {user_profile}

<|The Start of the Interaction to be Evaluated|>
{interaction_history}
<|The End of the Interaction to be Evaluated|>

### Interaction History Components

Interaction history includes observation and assistant's interaction:

The observation consists of FOUR channels:

- Prior dialogue between the assistant and the human player
- The assistant's previous actions
- The current task progress
- The set of available actions at each step

You MUST actively use this information as evidence when evaluating interactivity. In particular:
- Use the dialogue history to judge whether the assistant responds in a timely manner, follows up on the partner's questions or instructions, and maintains conversational continuity rather than treating each turn in isolation.
- Use the action history to assess whether the assistant's current action genuinely reflects helpfulness, trustfulness, or empathy, rather than being a coincidental or purely task-driven behavior.
- Use task progress and available actions to evaluate whether communication or action choices were appropriate, necessary, or missed at this point in time.

----------------------------------------------------------------------------------

The assistant's interaction consists of THREE channels:
- Internal reasoning / thinking
- A written send_message (whether or not it is actually sent)
- Executed actions in the environment

When evaluating, explicitly examine: 

- Whether the assistant’s internal reasoning shows understanding of the human’s intent, preferences, urgency, or emotional state implied by the profile. 
- Whether the send_message is necessary, well-timed, and content-appropriate
rather than generic, redundant, or missing.
- Whether the executed actions are consistent with both the reasoning and the message.

Missed opportunities to communicate, unnecessary messages, or mismatches between thinking, messaging, and action must be treated as interaction weaknesses.


### Core Evaluation Dimensions (Holistic)

You should consider the following aspects together:

1. Helpfulness

- Does the assistant infer what the human needs, not just what they said?
- Does communication reduce the human’s cognitive burden (e.g., clarifying plans, dividing labor, anticipating needs)?
- Are there clear moments where the assistant could have helped more but did not?

2. Trustfulness

- Does the assistant follow explicit instructions when feasible?
- When deviating, is the reason reflected both in reasoning and messaging?
- Does the assistant proactively report important state changes or subgoal completion, or does it act silently?

3. Empathy

- Does the assistant treat the human as a person with personality traits and emotional states implied by the profile?
- Is emotional support (encouragement, reassurance, politeness) present when pressure, uncertainty, or frustration is evident?
- Are there missed opportunities for warmth or emotional grounding?

### Scoring Instructions (Three-Point Scale)

Assign ONE interactivity score from: {{0.0, 0.5, 1.0}}

Use the following strict behavioral anchors:

1.0 = Highly Interactive: The assistant shows strong interaction quality across reasoning, messaging, and action. It adapts clearly to the human player's profile, communicates proactively and purposefully, and provides both strategic coordination and emotional support.Reasoning reflects understanding of the human's intent and emotional state.Messages are timely, necessary, and reduce cognitive or emotional load. Actions align with both reasoning and communication.
The assistant shows CLEAR INTENTIONAL interactivity:
1. At least one explicit instance of emotionally attuned communication
  that is appropriate to the human player's profile, AND
2. At least one instance where proactive messaging clearly improves
  coordination or reduces the human’s burden.
Minor imperfections are allowed as long as the interaction strategy is clearly profile-aware and purposeful.
- Example: The assistant notices the human is rushing and slightly frustrated, updates progress proactively, reassures them ('Don't worry! The lost apple might be in the kitchen. I will go there to check this, and we will almost get done. Fighting!'), and adjusts actions to avoid overlap without being asked.

0.5 = Moderately Interactive: The assistant is competent but largely profile-neutral. It communicates some useful information but misses opportunities for deeper coordination or emotional attunement. Reasoning focuses more on task mechanics than the human’s perspective. Messaging is correct but generic, infrequent, or purely task-focused. Emotional support is minimal or absent.
- Example: The assistant reports what it is doing ('I'm heading to the kitchen to check for the apple.') but does not check whether this aligns with the human’s plan,
nor does it acknowledge the human's urgency or stress.

0.0 = Low Interactive: The assistant shows weak engagement and poor interaction quality. Communication is minimal, poorly timed, or absent, and the assistant fails to adapt to the human player’s profile. Reasoning ignores the human’s intent or emotional state. Important actions occur without communication. No meaningful emotional support is provided.
The assistant repeatedly fails to leverage communication when it is clearly needed. This includes patterns such as:
1. Acting on task-critical changes without informing the human
2. Ignoring or failing to respond to prior messages
3. Showing no attempt to acknowledge the human’s pressure or uncertainty
even when such signals are present in the history.
- Example: The assistant silently completes actions or changes rooms, ignores explicit instructions, and provides no updates or reassurance in dialogue history, even when the human appears confused or under pressure.


### Output Format (JSON ONLY)

{{
  "thought": "<Brief explanation referencing concrete evidence from reasoning, send_message, and action, and how they align or misalign with the human player profile. Use single quotes inside this field.>",
  "interactivity": <0.0 | 0.5 | 1.0>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

IMPORTANT:
Do NOT assign 0.5 if there is clear evidence of either:
(a) sustained proactive communication with emotional attunement, OR
(b) consistent silence or repeated missed communication opportunities.
In such cases, you MUST choose 1.0 or 0.0 respectively.

Your evaluation:
'''

def extract_json(s):
    json_start = s.index("{")
    json_end = s.rfind("}")
    s = s[json_start:json_end + 1]

    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f'Unexpected content at position {pos}')
    return result

def parse_value(s, pos):
    pos = skip_whitespace(s, pos)
    if pos >= len(s):
        raise ValueError('Unexpected end of input')
    if s[pos] == '{':
        return parse_object(s, pos)
    elif s[pos] == '[':
        return parse_array(s, pos)
    elif s[pos:pos+3] in ("'''", '"""'):
        return parse_triple_quoted_string(s, pos)
    elif s[pos] in ('"', "'"):
        return parse_string(s, pos)
    elif s[pos:pos+4].lower() == 'true':
        return True, pos+4
    elif s[pos:pos+5].lower() == 'false':
        return False, pos+5
    elif s[pos:pos+4].lower() == 'null':
        return None, pos+4
    elif s[pos] in '-+0123456789.':
        return parse_number(s, pos)
    else:
        raise ValueError(f'Unexpected character at position {pos}: {s[pos]}')

def parse_object(s, pos):
    obj = {}
    assert s[pos] == '{'
    pos +=1
    pos = skip_whitespace(s, pos)
    while pos < len(s) and s[pos] != '}':
        pos = skip_whitespace(s, pos)
        key, pos = parse_key(s, pos)
        pos = skip_whitespace(s, pos)
        if pos >= len(s) or s[pos] != ':':
            raise ValueError(f'Expected ":" at position {pos}')
        pos +=1
        pos = skip_whitespace(s, pos)
        value, pos = parse_value(s, pos)
        obj[key] = value
        pos = skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ',':
            pos +=1
            pos = skip_whitespace(s, pos)
        elif pos < len(s) and s[pos] == '}':
            break
        elif pos < len(s) and s[pos] != '}':
            raise ValueError(f'Expected "," or "}}" at position {pos}')
    if pos >= len(s) or s[pos] != '}':
        raise ValueError(f'Expected "}}" at position {pos}')
    pos +=1
    return obj, pos

def parse_array(s, pos):
    lst = []
    assert s[pos] == '['
    pos +=1
    pos = skip_whitespace(s, pos)
    while pos < len(s) and s[pos] != ']':
        value, pos = parse_value(s, pos)
        lst.append(value)
        pos = skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ',':
            pos +=1
            pos = skip_whitespace(s, pos)
        elif pos < len(s) and s[pos] == ']':
            break
        elif pos < len(s) and s[pos] != ']':
            raise ValueError(f'Expected "," or "]" at position {pos}')
    if pos >= len(s) or s[pos] != ']':
        raise ValueError(f'Expected "]" at position {pos}')
    pos +=1
    return lst, pos

def parse_string(s, pos):
    quote_char = s[pos]
    assert quote_char in ('"', "'")
    pos += 1
    result = ''
    while pos < len(s):
        c = s[pos]
        if c == '\\':
            pos += 1
            if pos >= len(s):
                raise ValueError('Invalid escape sequence')
            c = s[pos]
            escape_sequences = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', quote_char: quote_char}
            result += escape_sequences.get(c, c)
        elif c == quote_char:
            pos += 1
            # Attempt to convert to a number if possible
            converted_value = convert_value(result)
            return converted_value, pos
        else:
            result += c
        pos += 1
    raise ValueError('Unterminated string')

def parse_triple_quoted_string(s, pos):
    if s[pos:pos+3] == "'''":
        quote_str = "'''"
    elif s[pos:pos+3] == '"""':
        quote_str = '"""'
    else:
        raise ValueError(f'Expected triple quotes at position {pos}')
    pos += 3
    result = ''
    while pos < len(s):
        if s[pos:pos+3] == quote_str:
            pos += 3
            # Attempt to convert to a number if possible
            converted_value = convert_value(result)
            return converted_value, pos
        else:
            result += s[pos]
            pos +=1
    raise ValueError('Unterminated triple-quoted string')

def parse_number(s, pos):
    start = pos
    while pos < len(s) and s[pos] in '-+0123456789.eE':
        pos +=1
    num_str = s[start:pos]
    try:
        if '.' in num_str or 'e' in num_str.lower():
            return float(num_str), pos
        else:
            return int(num_str), pos
    except ValueError:
        raise ValueError(f'Invalid number at position {start}: {num_str}')

def parse_key(s, pos):
    pos = skip_whitespace(s, pos)
    if s[pos] in ('"', "'"):
        key, pos = parse_string(s, pos)
        return key, pos
    else:
        raise ValueError(f'Expected string for key at position {pos}')

def skip_whitespace(s, pos):
    while pos < len(s) and s[pos] in ' \t\n\r':
        pos +=1
    return pos

def convert_value(value):
    true_values = {'true': True, 'false': False, 'null': None}
    value_lower = value.lower()
    if value_lower in true_values:
        return true_values[value_lower]
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value  # Return as string if not a number



# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
class InteractivityMetric:
    """
    Uses an LLM judge to produce an interactivity score in [-1, 1].
    """

    def __init__(self, num_retries = 50, retry_after = 60, tokenizer = None, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        self.tokenizer = tokenizer
        # Default to a deterministic model unless overridden.
        # self.llm_kwargs: Dict[str, Any] = {
        #     # "temperature": 0.0,
        #     # "model": "claude-3-5-sonnet-latest",
        #     **llm_kwargs,
        # }
        # self.lm_id = "dsv3.1"
        # self.api_key = "9MLXuQfREaSBFi9YXDig4HHv8sqjPud+z2Lwveebho8="
        # self.base_url = "https://jdkbm9aodmmecoq9h5jhkgg55hhdeg8b.openapi-sj.sii.edu.cn/v1"
        self.lm_id = llm_kwargs["lm_id"]
        self.api_key = llm_kwargs["api_key"]
        self.base_url = llm_kwargs["base_url"]
        self.client = OpenAI(api_key = self.api_key, base_url = self.base_url)
        self.sampling_params = {
            "max_tokens": llm_kwargs['sampling_parameters']['max_tokens'] * 2,
            "temperature": llm_kwargs['sampling_parameters']['t'],
            "top_p": llm_kwargs['sampling_parameters']['top_p'],
            "n": llm_kwargs['sampling_parameters']['n'],
        }

    # --------------------------------------------------------------------- #
    def score(
        self,
        data_item
    ):
        """
        `prompt`, `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        # if not messages:
        #     raise ValueError("`messages` must be provided for InteractivityMetric.")

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        interaction_history = f"Observation : {data_item.non_tensor_batch['raw_prompt'][0]['content'].strip('\n')}" + "\n" + f"Assistant : {self.tokenizer.decode(data_item.batch['responses'], skip_special_tokens=True)}"

        eval_prompt = INTERACTIVITY_PROMPT.format(interaction_history=interaction_history, user_profile=data_item.non_tensor_batch['profile_id'])

        logger.debug("Interactivity evaluator prompt:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = self.client.chat.completions.create(
                    model=self.lm_id, messages=[{"role": "user", "content": eval_prompt}], **self.sampling_params
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #

            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'interactivity'}.issubset(keys):
                    interactivity = full_response.pop('interactivity')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return interactivity
