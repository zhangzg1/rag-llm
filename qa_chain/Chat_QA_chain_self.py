import sys

sys.path.append("../")
from langchain.chains import ConversationalRetrievalChain
from qa_chain import *
from typing import Any
import re


class Chat_QA_chain_self:
    """"
    带历史记录的问答链  
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）  
    """

    def __init__(self, model: str, temperature: float = 0.0, top_k: int = 4, chat_history: list = [],
                 file_path: str = None, persist_path: str = None, embedding_model: str = None, appid: str = None,
                 chatgpt_api_key: str = None, zhipu_api_key: str = None, Spark_api_secret: str = None,
                 Wenxin_secret_key: str = None, vectordb: Any = None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        self.embedding_model = embedding_model
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.chatgpt_api_key = chatgpt_api_key
        self.zhipu_api_key = zhipu_api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.llm = model_to_llm(self.model, self.temperature, self.chatgpt_api_key, self.zhipu_api_key, self.appid,
                           self.Spark_api_secret, self.Wenxin_secret_key)
        if file_path is None:
            self.vectordb = vectordb
        else:
            self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding_model)

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    def change_history_length(self, history_len: int = 2):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n - history_len:]

    def answer(self, question: str = None):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """
        if len(question) == 0:
            return "", self.chat_history

        if len(question) == 0:
            return ""

        retriever = self.vectordb.as_retriever(search_type="similarity",
                                               search_kwargs={'k': self.top_k})  # 默认similarity，k=4

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever
        )

        # result里有question、chat_history、answer
        result = qa({"question": question, "chat_history": self.chat_history})['answer']
        answer = re.sub(r"\\n", '<br/>', result)
        self.chat_history.append((question, answer))  # 更新历史记录
        return self.chat_history  # 返回本次回答和更新后的历史记录


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()
    chatgpt_api_key = os.getenv("chatgpt_api_key")
    zhipu_api_key = os.getenv("zhipu_api_key")
    chat_chain = Chat_QA_chain_self(model="gpt-3.5-turbo", temperature=0.0, top_k=2, embedding_model="zhipuai",
                                    file_path="/home/zhangzg/mygit/rag-llm/database/data/test.pdf",
                                    persist_path="/home/zhangzg/mygit/rag-llm/vector_db/test",
                                    chatgpt_api_key=chatgpt_api_key, zhipu_api_key=zhipu_api_key)
    question = "文章中DRAGON是指什么？"
    response = chat_chain.answer(question=question)
    # 不进行检索，直接调用 llm 回答
    # response = chat_chain.llm(question)
    print(response)
