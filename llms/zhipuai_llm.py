from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI


# 继承自 langchain.llms.base.LLM
class ZhipuAILLM(LLM):
    model: str = "chatglm_std"  # 选择不同的模型
    temperature: float = 0.1
    zhipuai_api_key: str = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        def gen_glm_params(prompt):
            messages = [{"role": "user", "content": prompt}]
            return messages

        client = ZhipuAI(
            api_key=self.zhipuai_api_key
        )

        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"

    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        normal_params = {
            "temperature": self.temperature,
        }
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Zhipu"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {**{"model": self.model}, **self._default_params}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("zhipu_api_key")
    print(api_key)
    llm = ZhipuAILLM(model="chatglm_std", temperature=0, zhipuai_api_key=api_key)
    answer = llm("你是谁？")
    print(answer)
