from kg_retrieve import *
from qa_chain import *


class RAG_KG:
    def __init__(self, temperature=0.1, model: str = None, chatgpt_api_key: str = None, zhipu_api_key: str = None):
        self.entity = Question_Analyse()
        self.searcher = Entity_Search()
        self.llm = model_to_llm(model=model, temperature=temperature, chatgpt_api_key=chatgpt_api_key,
                                zhipu_api_key=zhipu_api_key)

    def rag_llm(self, question):
        # 根据用户问题，得到对应的实体
        entity = self.entity.question_to_entity(question)
        # 根据实体内容，检索得到实体的相关三元组信息
        triples = self.searcher.search_triple(entity, hop=1)
        # 将用户问题和检索到的三元组内容进行拼接，得到大模型的输入内容
        prompt = "这是一个关于用户的问题。给定以下知识三元组集合，三元组形式为(subject, relation, object)，表示subject和object之间存在relation关系" \
                 "请先从这些三元组集合中找到能够支撑问题的部分，在这里叫做证据。\n你现在有两个选择：1、如果三元组集合的内容为空，则直接回答下面的问题。" \
                 f"2、如果三元组集合的内容不为空，利用我给你提供的三元组信息，回答下面的问题。\n知识三元组集合为：{triples}" \
                 f"\n问题是：“{question}”\n请回答："
        print(prompt)
        # 调用大模型回答问题
        response = self.llm(prompt)
        return response


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()
    zhipu_api_key = os.getenv("zhipu_api_key")
    chatgpt_api_key = os.getenv("chatgpt_api_key")
    rag_kg = RAG_KG(model='chatglm_std', temperature=0.1, chatgpt_api_key=chatgpt_api_key, zhipu_api_key=zhipu_api_key)
    question = '你会干什么？'
    answer = rag_kg.rag_llm(question)
    # 不在知识图谱中进行检索，直接调用 llm 回答
    # answer = rag_kg.llm(question)
    print(answer)
