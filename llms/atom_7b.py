from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


class Atom_7b(LLM):
    tokenizer: Any = None
    model: Any = None
    temperature: float = None

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = max(0.1, min(1.0, temperature))
        self.tokenizer = AutoTokenizer.from_pretrained("xxxxxxxxxxxxx",
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("xxxxxxxxxxxxxx",
                                                          device_map="auto", trust_remote_code=True,
                                                          torch_dtype=torch.float16).eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any):
        input_prompt = f'<s>Human: {prompt}\n</s><s>Assistant: '
        input_ids = self.tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": self.temperature,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        generate_ids = self.model.generate(**generate_input)
        text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        assistant_response = re.search(r'Assistant: (.+)', text, re.DOTALL).group(1)
        return assistant_response

    @property
    def _llm_type(self) -> str:
        return "Atom-7b"
