import sys

sys.path.append("../")
from database import *


# 根据不同的embedding_api模型调用，返回不同的向量数据库
def get_vectordb(file_path: str = None, persist_path: str = None, embedding_model: str = None):
    """
    返回向量数据库对象
    输入参数：
    question：
    llms:
    vectordb:向量数据库(必要参数),一个对象
    template：提示模版（可选参数）可以自己设计一个提示模版，也有默认使用的
    embedding：可以使用zhipuai等embedding，不输入该参数则默认使用 openai embedding，注意此时api_key不要输错
    """
    if embedding_model == "zhipuai":
        vectordb = create_vectordb_zhipu(file_path, persist_path)
        return vectordb


if __name__ == '__main__':
    vectordb = get_vectordb(file_path="/home/zhangzg/mygit/rag-llm/database/data",
                            persist_path="/home/zhangzg/mygit/rag-llm/vector_db/all", embedding_model="zhipuai")
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
