import openai
from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff
from openai import OpenAI
# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "Qwen2-1.5B-Instruct": 8192,
}


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    """
    def __init__(self, 
                 role_messages, 
                 model="gpt-3.5-turbo-0301",
                 retrival_method="recent_k",
                 K=3,
                 agent_index=0,
                 agent_url=None,
                 agent_model=None,
                 agent_api_key=None,
                 player_url=None,
                 player_model=None,
                 player_api_key=None):
        '''
        args:  
        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        agent_url: API URL for agent (agent_index=0)
        agent_model: Model name for agent (agent_index=0)
        agent_api_key: API key for agent (agent_index=0)
        player_url: API URL for player (agent_index=1)
        player_model: Model name for player (agent_index=1)
        player_api_key: API key for player (agent_index=1)
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K
        self.agent_index = agent_index
        
        # Store API configuration for agent and player
        self.agent_url = agent_url
        self.agent_model = agent_model
        self.agent_api_key = agent_api_key
        self.player_url = player_url
        self.player_model = player_model
        self.player_api_key = player_api_key

        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        openai.api_key = key 
        rec = self.K  
        if trace == True: 
            self.K = 0 
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 
        response = "" 
        # print('\n\nmessages = \n\n{}\n\n'.format(messages))
        get_response = False
        retry_count = 0
        
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                return {}
            try:  
               
                prompt = convert_messages_to_prompt(messages)
                if self.agent_index == 0:
                    # Use agent configuration
                    if self.agent_url is None or self.agent_model is None or self.agent_api_key is None:
                        raise ValueError("agent_url, agent_model, and agent_api_key must be provided for agent_index=0")
                    
                    client = OpenAI(base_url=self.agent_url, api_key=self.agent_api_key)
                    response = client.chat.completions.create(
                        model=self.agent_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.0,
                        top_p=1.0,
                    )
                    time.sleep(1)
                else:
                    # Use player configuration
                    if self.player_url is None or self.player_model is None or self.player_api_key is None:
                        raise ValueError("player_url, player_model, and player_api_key must be provided for agent_index=1")
                    
                    client = OpenAI(base_url=self.player_url, api_key=self.player_api_key)
                    response = client.chat.completions.create(
                        model=self.player_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1.0,
                        top_p=1.0,
                    )
                    time.sleep(0.1)

                
                get_response = True

            except Exception as e:
                retry_count += 1
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(20)  
        return self.parse_response(response)

    def parse_response(self, response):
        if self.model == 'claude': 
            return response 
        elif self.model in ['text-davinci-003']:
            return response["choices"][0]["text"]
        elif self.model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314']:
            return response["choices"][0]["message"]["content"]
        else:
            return response.choices[0].message.content

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []