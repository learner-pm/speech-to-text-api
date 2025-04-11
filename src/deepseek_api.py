import os
import logging
from typing import Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class DeepSeekAPI:
    def __init__(self, api_key: Optional[str] = None):
        """初始化 DeepSeek API 客户端
        
        Args:
            api_key (str, optional): DeepSeek API 密钥。如果未提供，将尝试从环境变量获取。
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Please provide it or set DEEPSEEK_API_KEY environment variable.")
        
        # 初始化 OpenAI 客户端，设置 base_url 为 DeepSeek API 的地址
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def chat_completion(self, messages: list, model: str = "deepseek-chat", temperature: float = 0.7) -> Dict:
        """调用 DeepSeek 聊天完成 API
        
        Args:
            messages (list): 对话消息列表，每个消息包含 role 和 content
            model (str): 使用的模型名称
            temperature (float): 控制输出的随机性，范围 0-1
            
        Returns:
            Dict: API 响应结果
        """
        try:
            logger.info(f"Calling DeepSeek API with model: {model}")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            # 将响应转换为字典格式
            return {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            raise

    def text_embedding(self, text: str, model: str = "deepseek-embedding") -> Dict:
        """调用 DeepSeek 文本嵌入 API
        
        Args:
            text (str): 要生成嵌入的文本
            model (str): 使用的模型名称
            
        Returns:
            Dict: API 响应结果
        """
        try:
            logger.info(f"Calling DeepSeek Embedding API with model: {model}")
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            
            # 将响应转换为字典格式
            return {
                "object": response.object,
                "data": [
                    {
                        "object": item.object,
                        "embedding": item.embedding,
                        "index": item.index
                    }
                    for item in response.data
                ],
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error calling DeepSeek Embedding API: {str(e)}")
            raise 