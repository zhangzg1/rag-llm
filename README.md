# rag-llm

## 1、介绍

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的 AI 技术，通过检索外部知识库中的相关信息，增强语言模型的生成能力。本项目设计的 RAG 系统支持两种类型的外挂数据：文本数据和图数据，旨在充分利用不同类型数据的优势，提升系统对复杂问题的理解和回答能力。

当外挂知识库的数据是文本数据时，可以支持上传的数据是 pdf 文件和 md 文档等等。当外挂知识库的数据是图数据时，主要是连接 neo4j 数据库，然后将数据存储在里面，最后进行检索生成。

## 2、下载源码

```
git clone https://github.com/zhangzg1/rag-llm.git
cd rag-llm
```

## 3、安装依赖环境（Linux）

```
# 创建虚拟环境
conda create -n rag-llm python=3.10
conda activate rag-llm
# 安装其他依赖包
pip install -r requirements.txt
```

## 4、核心架构

![image](https://github.com/zhangzg1/rag-llm/blob/main/database/figures/rag.jpg)

整个 RAG 项目的核心模块为三个部分：

1、创建向量数据库：对数据进行一系列处理后，再将其文本向量化并存储到向量数据库中

2、检索增强：将用户的问题输入到检索系统中，从数据库中检索相关信息，并对检索到的信息进行处理和增强。

3、LLM 生成：将增强后的信息输入到生成模型中，LLM 根据这些信息生成答案。

## 5、运行测试

```
# 基于文本的RAG，运行后会启动web界面
python run_gradio.py

# 基于知识图谱的RAG
python rag_with_kg.py
```
