from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import transformers
import torch
import re


class Llama3_8b(LLM):
    pipeline: Any = None
    temperature: float = None

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="/home/zhangzg/LLM/model/Llama3-Chinese-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
        self.temperature = max(0.1, min(1.0, temperature))

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any):
        messages = [{"role": "user", "content": prompt}]
        prompts = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipeline(
            prompts,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9
        )
        text = outputs[0]["generated_text"][len(prompt):]
        assistant_response = re.search(r'assistant<\|end_header_id\|>(.*?)$', text, re.DOTALL).group(1).strip()
        return assistant_response

    @property
    def _llm_type(self) -> str:
        return "Llama3-8b"


if __name__ == "__main__":
    llm = Llama3_8b(temperature=0.3)
    answer = llm("你是谁？")
    print(answer)
