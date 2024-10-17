from py2neo import Graph
import re


class Entity_Search:
    def __init__(self):
        self.graph = Graph(
            host="xxx.xx.xxx.xxx",
            user="neo4j",
            password="neo4j")

    # 从 neo4j 查询后得到了该节点的所有 1 跳连接关系，然后进行处理得到三元组
    def get_triple_one(self, query):
        triples = []
        for rel in query[0]['relationships']:
            str_rel = str(rel)
            match = re.match(r'\((\w+)\)-\[:(\w+)\s+{name:\s*\'Edge\'}\]->\((\w+)\)', str_rel)
            if match:
                # 提取三元组中的元素
                start_node = match.group(1)
                relationship = match.group(2)
                end_node = match.group(3)
                # 创建三元组
                triple = [start_node, relationship, end_node]
                triples.append(triple)
        return triples

    # 从 neo4j 查询后得到了该节点的所有 2 跳的连接关系，然后进行处理得到三元组
    def get_triple_two(self, query):
        triples = []
        for rel in query[0]['relationships']:
            for item in rel:
                str_rel = str(item)
                match = re.match(r'\((\w+)\)-\[:(\w+)\s+{name:\s*\'Edge\'}\]->\((\w+)\)', str_rel)
                if match:
                    start_node = match.group(1)
                    relationship = match.group(2)
                    end_node = match.group(3)
                    triple = [start_node, relationship, end_node]
                    triples.append(triple)
        # 使用set来去重，同时将内部的列表转换为元组
        unique_triples = list(set(tuple(item) for item in triples))
        # 后续可以将得到的所有 2 跳内的三元组处理成一些推理路径
        return unique_triples

    # 根据问题 query 中的实体从知识图谱中检索出相关的三元组信息
    def search_triple(self, entity_list, hop):
        all_triples = []
        for entity in entity_list:
            # 根据跳数将检索到的信息进行处理，得到三元组
            if hop == 1:
                # 从数据库中查询该节点的所有的 1 跳连接关系。
                query_one = f"MATCH (n)-[r]-(neighbor) WHERE n.name = '{entity}' RETURN collect(r) as relationships"
                query = self.graph.run(query_one).data()
                triples = self.get_triple_one(query)
                for item in triples: all_triples.append(item)
            elif hop == 2:
                # 从数据库中查询该节点的所有的 2 跳连接关系。
                query_two = f"MATCH (n)-[r*1..2]-(neighbor) WHERE n.name = '{entity}' RETURN collect(r) as relationships"
                query = self.graph.run(query_two).data()
                triples = self.get_triple_two(query)
                for item in triples: all_triples.append(item)
            else:
                all_triples = []
        # 使用set来去重，同时将内部的列表转换为元组
        unique_triples = list(set(tuple(item) for item in all_triples))
        return unique_triples
