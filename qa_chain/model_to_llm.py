import sys

sys.path.append("../")
from llm.wenxin_llm import Wenxin_LLM
from llm.spark_llm import Spark_LLM
from llm.zhipuai_llm import ZhipuAILLM
# from langchain.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key
from llm.atom_7b import Atom_7b
from llm.llama3_8b import Llama3_8b


# 调用不同模型的基础模型或 api_key 来进行对话
def model_to_llm(model: str = None, temperature: float = 0.0, api_key: str = None, appid: str = None,
                 Spark_api_secret: str = None, Wenxin_secret_key: str = None):
    # 调用 GPT 模型
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        if api_key == None:
            api_key = parse_llm_api_key("openai")
        # llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)
    # 调用百度文心大模型
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        if api_key == None or Wenxin_secret_key == None:
            api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
        llm = Wenxin_LLM(model=model, temperature=temperature, api_key=api_key, secret_key=Wenxin_secret_key)
    # 调用星火大模型
    elif model in ["Spark-1.5", "Spark-2.0"]:
        if api_key == None or appid == None and Spark_api_secret == None:
            api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
        llm = Spark_LLM(model=model, temperature=temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
    # 调用 ChatGLM 模型
    elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
        if api_key == None:
            api_key = parse_llm_api_key("zhipuai")
        llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature=temperature)
    # 调用开源的 Llama 中文模型
    elif model in ["Atom-7b", "Llama3-8b"]:
        if model == "Atom-7b":
            llm = Atom_7b(temperature=temperature)
        if model == "Llama3-8b":
            llm = Llama3_8b(temperature=temperature)
    else:
        raise ValueError(f"model{model} not support!!!")
    return llm


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("zhipu_api_key")
    llm = model_to_llm(model="Atom-7b", temperature=0.0, api_key=api_key, appid=None,
                       Spark_api_secret=None, Wenxin_secret_key=None)
    answer = llm("你是谁？")
    print(answer)
