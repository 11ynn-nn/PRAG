import chromadb
from collections import defaultdict

def search_structured_routing(query_vector, db_path="servers_data/leader-a/chroma_db"):
    """
    双层检索并返回结构化字典: { "device": ["entity1", "entity2", ...] }
    """
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        domain_collection = client.get_collection(name="domain_routing_vdb")
        entity_collection = client.get_collection(name="entity_routing_vdb")
    except Exception as e:
        return {"error": f"Collection access failed: {e}"}

    # --- 1. 检索 Top-5 Domains ---
    domain_results = domain_collection.query(
        query_embeddings=[query_vector],
        n_results=5
    )
    top_domains = [meta['domain_name'] for meta in domain_results['metadatas'][0]]
    
    if not top_domains:
        return {}

    # --- 2. 在 Domain 范围内检索 Top-30 实体 ---
    entity_results = entity_collection.query(
        query_embeddings=[query_vector],
        n_results=30,
        where={"domain": {"$in": top_domains}}
    )

    # --- 3. 结构化梳理 (按设备归类) ---
    # 使用 defaultdict(list) 可以方便地向同一个 key 添加多个 value
    structured_data = defaultdict(list)
    
    metadatas = entity_results['metadatas'][0]
    for meta in metadatas:
        device = meta.get('source_edge', 'UnknownDevice')
        entity = meta.get('entity_name', 'UnknownEntity')
        
        # 将实体加入对应设备的列表，并顺便去重
        if entity not in structured_data[device]:
            structured_data[device].append(entity)

    # 转换回普通字典返回
    return dict(structured_data)

# === 模拟输出效果 ===
if __name__ == "__main__":
    # 模拟一个 768 维的向量
    import numpy as np
    mock_vector = np.random.rand(768).tolist()
    
    result_dict = search_structured_routing(mock_vector)
    
    import json
    print("📊 结构化检索结果:")
    print(json.dumps(result_dict, indent=4, ensure_ascii=False))

    # 后续提取示例：
    # for edge, entities in result_dict.items():
    #     print(f"设备 {edge} 发现了 {len(entities)} 个相关实体")