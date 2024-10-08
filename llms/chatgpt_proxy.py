from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import json
import requests


class ChatGPT_Proxy(LLM):
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    chatgpt_api_key: str = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        url = "xxxxxxxxxxxxxx"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.chatgpt_api_key
        }
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            res = response.json()
            return res['choices'][0]['message']['content']
        except requests.exceptions.RequestException as err:
            print("Request error occurred:", err)
            return None

    @property
    def _llm_type(self) -> str:
        return "chatgpt"
