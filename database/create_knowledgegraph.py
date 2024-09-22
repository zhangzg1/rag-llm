import json
from py2neo import Graph, Node


class Knowledge_Graph:
    def __init__(self, data_path, host="xxx.xx.xxx.xxx", username="neo4j", password="neo4j"):
        self.data_path = data_path
        self.graph = Graph(host=host, auth=(username, password))

    # 从 json 文件中读取数据，提取数据中节点实体和关系边的信息，返回所有节点实体集合和边关系字典。
    def read_nodes(self):
        all_json = []
        head_node, tail_node = list(), list()
        relations = {'point_to': []}
        with open(self.data_path, 'r') as file:
            data_str = file.read()
            all_data = json.loads(data_str)
            for data_json in all_data:
                head_node.append(data_json.get('head_node', ''))
                tail_node.append(data_json.get('tail_node', ''))
                relations['point_to'].append([data_json['head_node'], data_json['tail_node'], data_json['relation']])
                all_json.append(data_json)
        return head_node, relations, tail_node, all_json


    # 在 neo4j 图数据库中创建普通节点，该节点通常只有一个属性，node_data 是一个节点集合的数据。
    def create_node(self, label, node_data):
        for node_name in node_data:
            query = f"MATCH (n:{label}) WHERE n.name = '{node_name}' RETURN n"
            result = self.graph.evaluate(query)
            if not result:
                node = Node(label, name=node_name)
                self.graph.create(node)

    def create_relationship(self, start_label, end_label, relations, rel_type):
        """
        在 neo4j 图数据库中创建实体间的关系，并添加关系名称。
        start_label: 起始节点的类型。
        end_label: 终点节点的类型。
        relations: 包含起始和终点实体名称的关系列表。
        rel_type: 创建的边关系类型。
        """
        for start, end, middle in relations:
            query = f"MATCH (a:{start_label}), (b:{end_label}) " \
                    f"WHERE a.name='{start}' AND b.name='{end}' " \
                    f"CREATE (a)-[r:{middle} {{name: '{rel_type}'}}]->(b)"
            try:
                self.graph.run(query)
            except Exception as e:
                print(f"Failed to create relationship: {e}")

    # 创建知识图谱，将数据放到所创建的节点和边中。
    def create_graph(self):
        head_node, relations, tail_node, all_json = self.read_nodes()
        self.create_node("Node", head_node)
        self.create_node("Node", tail_node)
        self.create_relationship("Node", "Node", relations['point_to'], "Edge")


if __name__ == '__main__':
    data_path = '/database/data/kg.json'
    kg = Knowledge_Graph(data_path)
    print("导入知识图谱...")
    kg.create_graph()
    print("知识图谱导入完成.")
