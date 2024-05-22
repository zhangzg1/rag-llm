from py2neo import Graph
import ahocorasick


class Question_Analyse:
    def __init__(self):
        self.graph = Graph(
            host="172.22.109.201",
            user="neo4j",
            password="neo4j")

    # 获取知识图谱中所有节点内容
    def get_all_nodes(self):
        query = "MATCH (n) RETURN n.name AS name"
        query_result = self.graph.run(query)
        node_list = []
        # 遍历查询结果
        for node in query_result:
            node_list.append(node["name"])
        return node_list

    # 使用 Aho-Corasick 算法，用于在文本中高效地查找多个关键词。
    def initialize_automaton(self, keywords):
        # 创建Aho-Corasick自动机
        A = ahocorasick.Automaton()
        for idx, key in enumerate(keywords):
            A.add_word(key.lower(), (idx, key))
        A.make_automaton()
        return A

    def question_to_entity(self, question):
        question_lower = question.lower()
        # 搜索关键词，使用 set 去重
        all_entity = set()
        node_list = self.get_all_nodes()
        automaton = self.initialize_automaton(node_list)
        for end_index, (idx, original_value) in automaton.iter(question_lower):
            all_entity.add(original_value)
        return list(all_entity)


if __name__ == '__main__':
    Q = Question_Analyse()
    question = '请介绍一下台湾的历史文化。'
    entity = Q.question_to_entity(question)
    print(entity)
