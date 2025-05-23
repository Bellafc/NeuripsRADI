#!/usr/bin/env python3

def remove_duplicates(graph):
    """
    从 graph 字典中移除 nodes 和 edges 列表中的重复项：
      - 对于 nodes，依据节点的 "id" 去重，保留后出现的那个；
      - 对于 edges，依据 ("from_id", "to_id") 组合去重，保留后出现的那个。
    """
    # 处理 nodes：反向遍历，第一次遇到的节点为保留项
    seen_ids = set()
    new_nodes = []
    for node in reversed(graph.get("nodes", [])):
        node_id = node.get("id")
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            new_nodes.append(node)
    graph["nodes"] = list(reversed(new_nodes))
    
    # 处理 edges：反向遍历，依据 (from_id, to_id) 组合去重
    seen_edges = set()
    new_edges = []
    for edge in reversed(graph.get("edges", [])):
        key = (edge.get("from_id"), edge.get("to_id"))
        if key not in seen_edges:
            seen_edges.add(key)
            new_edges.append(edge)
    graph["edges"] = list(reversed(new_edges))
    
    return graph

def sort_graph(graph):
    """
    对图数据进行排序：
      - nodes 列表根据 "id" 从小到大排序
      - edges 列表根据 "from_id" 从小到大排序
        （如果需要，可对 "to_id" 做进一步排序）
    """
    graph["nodes"] = sorted(graph.get("nodes", []), key=lambda node: node.get("id", 0))
    graph["edges"] = sorted(graph.get("edges", []), key=lambda edge: (edge.get("from_id", 0), edge.get("to_id", 0)))
    return graph

def main():
    # 示例数据，包含重复的节点和边
    graph = {
        "nodes": [
            {"id": 45, "class_name": "walllamp", "states": []},
            {"id": 34, "class_name": "barsoap", "states": []},
            {"id": 45, "class_name": "walllamp_duplicate", "states": []},  # 重复 id=45，保留后面的
            {"id": 24, "class_name": "toilet", "states": ["CLOSED", "CLEAN"]}
        ],
        "edges": [
            {"from_id": 1, "relation_type": "CLOSE", "to_id": 36},
            {"from_id": 1, "relation_type": "INSIDE", "to_id": 11},
            {"from_id": 1, "relation_type": "CLOSE", "to_id": 36},  # 重复边 (1,36)，保留后面的
            {"from_id": 1, "relation_type": "INSIDE", "to_id": 11}   # 重复边 (1,11)，保留后面的
        ]
    }
    
    print("原始数据：")
    print("Nodes:")
    for node in graph["nodes"]:
        print(node)
    print("Edges:")
    for edge in graph["edges"]:
        print(edge)
    
    # 去除重复项
    remove_duplicates(graph)
    
    # 对整个字典排序
    sort_graph(graph)
    
    print("\n整理后的数据：")
    print("Nodes (按 id 升序排序):")
    for node in graph["nodes"]:
        print(node)
    print("\nEdges (按 from_id 升序排序):")
    for edge in graph["edges"]:
        print(edge)

if __name__ == "__main__":
    main()
