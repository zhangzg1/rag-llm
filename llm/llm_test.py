from zhipuai_llm import ZhipuAILLM
from atom_7b import Atom_7b

def zhipuai_model(question, api_key):
    model = ZhipuAILLM(
        model="chatglm_std",
        temperature=0,
        zhipuai_api_key=api_key
    )
    answer = model(question)
    return answer

def atom(question):
    model = Atom_7b()
    answer = model(question)
    return answer

if __name__ == "__main__":
    api_key = "bde889da0225163afdd5cc6dbd12cd7d.PTusGc2TFVCTZeQb"
    prompt = "你是谁？"
    answer = zhipuai_model(prompt, api_key)
    # answer = atom(prompt)
    print(answer)

