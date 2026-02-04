# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Optional
import ray
import asyncio
import time
import threading
from dataclasses import dataclass
from collections import deque
from openai import AsyncOpenAI

@dataclass
class BatchRequest:
    """单个推理请求"""
    request_id: str
    messages: List[Dict[str, str]]
    sampling_params: Dict[str, Any]
    future: asyncio.Future
    model: str
    stop: Optional[str] = None

class GlobalBatchScheduler:
    """
    全局批处理调度器，所有 LLM 实例共享
    单例模式确保整个 Worker 只有一个调度器
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, base_url: str = None, lm_id: str = None,
                 max_batch_size: int = 8, max_wait_time: float = 0.5):
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
        
        self.base_url = base_url
        self.lm_id = lm_id
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        
        self.request_queue: deque[BatchRequest] = deque()
        self.batch_task: Optional[asyncio.Task] = None
        self.last_batch_time = time.time()
        self._initialized = True
        self._processing = False
    
    async def generate(self, messages: List[Dict[str, str]], 
                      sampling_params: Dict[str, Any],
                      model: str,
                      stop: Optional[str] = None,
                      api_key: str = "None") -> str:
        """异步生成接口"""
        request_id = f"req_{id(messages)}_{time.time()}"
        future = asyncio.Future()
        request = BatchRequest(
            request_id=request_id,
            messages=messages,
            sampling_params=sampling_params,
            future=future,
            model=model,
            stop=stop
        )
        
        # 加入队列
        self.request_queue.append(request)
        
        # 检查是否需要立即处理
        if len(self.request_queue) >= self.max_batch_size and not self._processing:
            asyncio.create_task(self._process_batch(api_key))
        elif self.batch_task is None or self.batch_task.done():
            # 启动定时批处理任务
            self.batch_task = asyncio.create_task(self._wait_and_process(api_key))
        
        # 等待结果
        return await future
    
    async def _wait_and_process(self, api_key: str):
        """等待一段时间后处理批次"""
        try:
            await asyncio.sleep(self.max_wait_time)
            if len(self.request_queue) > 0 and not self._processing:
                await self._process_batch(api_key)
        except Exception as e:
            print(f"Error in _wait_and_process: {e}")
    
    async def _process_batch(self, api_key: str):
        """处理当前批次的所有请求"""
        if len(self.request_queue) == 0 or self._processing:
            return
        
        self._processing = True
        
        try:
            # 取出当前批次
            batch_size = min(len(self.request_queue), self.max_batch_size)
            batch_requests = [self.request_queue.popleft() for _ in range(batch_size)]

            # 添加详细日志
            print(f"[BatchScheduler-{id(self)}] Processing batch of {batch_size} requests at {time.time()}")
            print(f"[BatchScheduler-{id(self)}] Request IDs: {[req.request_id for req in batch_requests]}")
            
            # 构造批量请求
            client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
            
            # 并发发送所有请求（vLLM 会在服务端做批处理）
            tasks = []
            for req in batch_requests:
                # 处理 Qwen 模型的特殊格式
                if 'wen' in req.model.lower() or 'qwen' in req.model.lower():
                    from .utils import convert_messages_to_prompt
                    prompt = convert_messages_to_prompt(req.messages)
                    task = client.chat.completions.create(
                        model=req.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=req.sampling_params.get('temperature', 0.7),
                        top_p=req.sampling_params.get('top_p', 0.85),
                        stop=req.stop,
                        max_tokens=req.sampling_params.get('max_tokens', 256)
                    )
                else:
                    # 对于其他模型，使用 messages 直接
                    task = client.chat.completions.create(
                        model=req.model,
                        messages=req.messages,
                        stop=req.stop,
                        temperature=req.sampling_params.get('temperature', 0.0),
                        max_tokens=req.sampling_params.get('max_tokens', 256)
                    )
                tasks.append(task)
            
            # 等待所有响应
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 将结果返回给各个 future
            for req, response in zip(batch_requests, responses):
                if isinstance(response, Exception):
                    req.future.set_exception(response)
                else:
                    req.future.set_result(response.choices[0].message.content)
                    
        except Exception as e:
            print(f"[BatchScheduler] Error processing batch: {e}")
            # 如果批处理失败，通知所有请求
            for req in batch_requests:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            self._processing = False
            self.last_batch_time = time.time()


class ProAgentBatchClientWorker:
    """
    Ray remote actor that holds a batch scheduler for ProAgent P1 LLM calls.
    Each worker manages batch scheduling for one environment's P1 agent.
    """
    
    # 类级别的调度器，所有该 Worker 的 agent 共享
    _batch_scheduler = None
    _scheduler_lock = threading.Lock()

    def __init__(self, env_id: int, base_url: str, lm_id: str, 
                 max_batch_size: int = 8, max_wait_time: float = 0.5):
        self.env_id = env_id
        self.base_url = base_url
        self.lm_id = lm_id
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        # 初始化全局调度器（只初始化一次）
        with ProAgentBatchClientWorker._scheduler_lock:
            if ProAgentBatchClientWorker._batch_scheduler is None:
                print(f"[Worker-{env_id}] Initializing GlobalBatchScheduler")
                ProAgentBatchClientWorker._batch_scheduler = GlobalBatchScheduler(
                    base_url=base_url,
                    lm_id=lm_id,
                    max_batch_size=max_batch_size,
                    max_wait_time=max_wait_time
                )
    
    def generate(self, messages: List[Dict[str, str]], 
                 sampling_params: Dict[str, Any],
                 model: str,
                 stop: Optional[str] = None,
                 api_key: str = "None") -> str:
        """同步包装器，调用异步生成（Ray remote method）"""
        # This is a Ray remote method, so it will be called remotely
        # We need to create an event loop in the Ray worker
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 如果事件循环正在运行，使用 nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                print("Warning: nest_asyncio not installed, may cause issues")
            return loop.run_until_complete(
                ProAgentBatchClientWorker._batch_scheduler.generate(
                    messages, sampling_params, model, stop, api_key
                )
            )
        else:
            return loop.run_until_complete(
                ProAgentBatchClientWorker._batch_scheduler.generate(
                    messages, sampling_params, model, stop, api_key
                )
            )


_global_batch_client_actor: Optional[ray.actor.ActorHandle] = None
_global_batch_client_lock = threading.Lock()


def get_global_batch_client(base_url: str,
                            lm_id: str,
                            max_batch_size: int = 8,
                            max_wait_time: float = 0.5,
                            force_new: bool = False) -> ray.actor.ActorHandle:
    """Return a singleton batch client actor shared across environments."""
    global _global_batch_client_actor
    with _global_batch_client_lock:
        if force_new and _global_batch_client_actor is not None:
            try:
                ray.kill(_global_batch_client_actor)
            except Exception:
                pass
            _global_batch_client_actor = None
        if _global_batch_client_actor is None:
            BatchClient = ray.remote(ProAgentBatchClientWorker)
            _global_batch_client_actor = BatchClient.remote(
                env_id=0,
                base_url=base_url,
                lm_id=lm_id,
                max_batch_size=max_batch_size,
                max_wait_time=max_wait_time,
            )
        return _global_batch_client_actor
