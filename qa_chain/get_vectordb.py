import sys

sys.path.append("../")
from database import *


# 根据不同的embedding_api模型调用，返回不同的向量数据库
def get_vectordb(file_path: str = None, persist_path: str = None, embedding_model: str = None):
    """
    返回向量数据库对象
    """
    if embedding_model == "zhipuai":
        vectordb = create_vectordb_zhipu(file_path, persist_path)
        return vectordb
