from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class Wenxin_LLM(LLM):
    model: str = None
    temperature: float = None
    api_key: str = None
    secret_key: str = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        pass

    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
        }
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Wenxin"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**{"model": self.model}, **self._default_params}
