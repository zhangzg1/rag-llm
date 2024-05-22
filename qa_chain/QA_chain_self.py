import sys

sys.path.append("../")
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from qa_chain import *
from typing import Any
import re


class QA_chain_self():
    """"
    不带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火需要输入
    - api_key：所有模型都需要
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型  
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq    
    """

    # 基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    default_template_rq = """请根据检索到的内容来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。检索内容: {context}\
问题: {question}
你的回答: """

    def __init__(self, model: str, temperature: float = 0.0, top_k: int = 4, file_path: str = None,
                 persist_path: str = None, embedding_model: str = None, appid: str = None, chatgpt_api_key: str = None,
                 zhipu_api_key: str = None, Spark_api_secret: str = None, Wenxin_secret_key: str = None,
                 template=default_template_rq, vectordb: Any = None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.chatgpt_api_key = chatgpt_api_key
        self.zhipu_api_key = zhipu_api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding_model = embedding_model
        self.template = template
        if file_path is None:
            self.vectordb = vectordb
        else:
            self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding_model)
        self.llm = model_to_llm(self.model, self.temperature, self.chatgpt_api_key, self.zhipu_api_key, self.appid,
                                self.Spark_api_secret, self.Wenxin_secret_key)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                              template=self.template)
        self.retriever = self.vectordb.as_retriever(search_type="similarity",
                                                    search_kwargs={'k': self.top_k})  # 默认similarity，k=4
        # 自定义 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                    retriever=self.retriever,
                                                    return_source_documents=True,
                                                    chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT})

    # 基于大模型的问答 prompt 使用的默认提示模版
    # default_template_llm = """请回答下列问题:{question}"""

    def answer(self, question: str = None):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """
        if len(question) == 0:
            return ""

        result = self.qa_chain({"query": question})["result"]
        answer = re.sub(r"\\n", '<br/>', result)
        return answer


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    chatgpt_api_key = os.getenv("chatgpt_api_key")
    zhipu_api_key = os.getenv("zhipu_api_key")
    chain = QA_chain_self(model="chatglm_std", temperature=0.0, top_k=4, embedding_model="zhipuai",
                          file_path="/home/zhangzg/mygit/rag-llm/database/data/test.pdf",
                          persist_path="/home/zhangzg/mygit/rag-llm/vector_db/test",
                          chatgpt_api_key=chatgpt_api_key, zhipu_api_key=zhipu_api_key)
    question = "文章中DRAGON是指什么？"
    response = chain.answer(question=question)
    # 不进行检索，直接调用 llm 回答
    # response = chain.llm(question)
    print(response)
