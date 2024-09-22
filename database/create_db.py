import sys

sys.path.append("../")
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
from tqdm import tqdm


# 获取文件路径函数
def get_files(dir_path):
    file_list = []
    for filepath, dir_names, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
    return file_list


# 加载文件函数
def get_text(dir_or_file_path):
    docs = []
    if os.path.isdir(dir_or_file_path):
        file_lst = get_files(dir_or_file_path)
        for one_file in tqdm(file_lst):
            file_type = one_file.split('.')[-1]
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)
            elif file_type == 'pdf':
                loader = PyMuPDFLoader(one_file)
            else:
                continue
            docs.extend(loader.load())
    elif os.path.isfile(dir_or_file_path):
        file_type = dir_or_file_path.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(dir_or_file_path)
        elif file_type == 'pdf':
            loader = PyMuPDFLoader(dir_or_file_path)
        else:
            raise ValueError("Unsupported file type.")
        docs.extend(loader.load())
    else:
        raise FileNotFoundError("The specified path does not exist.")
    return docs


# 对文本内容进行分块处理
def text_spilt(docs, chunk_size=500, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


# 加载文件夹或目标文件中的文本数据
def load_docs(tar_path):
    docs = []
    tar_path_all = [tar_path]
    for path in tar_path_all:
        docs.extend(get_text(path))
    return docs


# 基于 zhipuai 创建向量数据库
def create_vectordb_zhipu(tar_path, persist_directory):
    # 加载文件夹或目标文件中的文本数据
    docs = load_docs(tar_path)
    # 对文本内容进行分块处理
    split_docs = text_spilt(docs)
    # 使用智谱AI Embedding
    embeddings = ZhipuAIEmbeddings()
    # 将文本数据向量化并存储到向量数据库中
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()
    return vectordb

