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
                 base_url=None,
                 api_key=None,
                 batch_client=None):
        '''
        args:  
        use_similarity: 
        dia_num: the num of dia use need retrival from dialog history
        base_url: Base URL for OpenAI-compatible API
        api_key: API key for OpenAI-compatible API
        batch_client: Ray remote batch client for async batching (optional)
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K
        self.base_url = base_url
        self.api_key = api_key or "EMPTY"
        self.batch_client = batch_client  # Ray remote batch client

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
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 
        response = "" 
        # print('\n\nmessages = \n\n{}\n\n'.format(messages))
        get_response = False
        retry_count = 0
        
        # If batch_client is available, use async batching
        if self.batch_client is not None and self.base_url:
            try:
                # Prepare sampling parameters
                sampling_params = {
                    'temperature': temperature if temperature > 0 else 0.7,
                    'max_tokens': 256
                }
                if 'wen' in self.model.lower() or 'qwen' in self.model.lower():
                    sampling_params['top_p'] = 0.85
                
                # Use batch client for async batching (Ray remote call)
                import ray
                response_text = ray.get(self.batch_client.generate.remote(
                    messages=messages,
                    sampling_params=sampling_params,
                    model=self.model,
                    stop=stop,
                    api_key=self.api_key
                ))
                # Create a mock response object for parse_response
                class MockResponse:
                    def __init__(self, content):
                        self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})()]
                response = MockResponse(response_text)
                return self.parse_response(response)
            except Exception as e:
                rprint(f"[yellow][WARNING][/yellow]: Batch client failed, falling back to direct API: {e}")
                # Fall through to direct API call
        
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                # Return a fallback response in the expected format (l2-ap_merged)
                return """<think>
Query failed after 3 retries. Using default wait action to avoid blocking the game.
</think>
<message>
</message>
<action>
wait(1)
</action>"""
            try:  
                # Use OpenAI-compatible client if base_url is provided
                if self.base_url:
                    client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                    if 'wen' in self.model.lower() or 'qwen' in self.model.lower():
                        # For Qwen models, convert messages to prompt
                        prompt = convert_messages_to_prompt(messages)
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature if temperature > 0 else 0.7,
                            top_p=0.85,
                            stop=stop,
                            max_tokens=256
                        )
                    else:
                        # For other models, use messages directly
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            stop=stop,
                            temperature=temperature,
                            max_tokens=256
                        )
                    time.sleep(1)
                elif self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages) 
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens = 256
                    )
                    time.sleep(10)  
                elif 'gpt' in self.model:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=temperature, 
                        max_tokens = 256
                    )
                    time.sleep(10) 
                else:
                    prompt = convert_messages_to_prompt(messages)
                    # base_url from env.proagent.base_url (passed via Module __init__)
                    client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.8,
                        top_p=0.85,
                    )
                    time.sleep(1)

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
        print(f'[Module] 📊 当前 token 数量: {self.prompt_token_length} / {limit}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'[Module] 🔄 移除旧消息后 token 数量: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []