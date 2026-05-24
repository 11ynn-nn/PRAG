import json
import networkx as nx
import igraph as ig
import leidenalg
import os
import requests  # 新增：用于调用 Ollama API
import time

# ================= 配置部分 =================
DATA_DIR = "servers_data\edge-a5"

# 确认你的 LightRAG 输出文件名，通常是 relationships (带hip)，如果你的文件确实是 relations 请修改这里
FILES = {
    "graph": "graph_chunk_entity_relation.graphml",
    "entities": "kv_store_full_entities.json",
    "relationships": "kv_store_full_relations.json" 
}

# Ollama 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

# ================= 1. Ollama 调用函数 (新增) =================
def call_ollama_summary(prompt, model=MODEL_NAME):
    """
    调用本地 Ollama 接口生成摘要
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # 关闭流式输出，一次性获取结果
        "options": {
            "temperature": 0.3,   # 较低温度，保证摘要准确性
            "num_ctx": 8192       # 确保上下文窗口足够大，Qwen2.5 支持较长上下文
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return "[Error: SLM generation failed]"

# ================= 2. 数据加载函数 =================
def load_lightrag_data(base_dir):
    print(f"Loading data from {base_dir}...")
    
    # 路径拼接检查
    entity_path = os.path.join(base_dir, FILES["entities"])
    rel_path = os.path.join(base_dir, FILES["relationships"])
    graph_path = os.path.join(base_dir, FILES["graph"])

    if not os.path.exists(entity_path):
        raise FileNotFoundError(f"Entities file not found: {entity_path}")

    # 1. 加载实体
    with open(entity_path, 'r', encoding='utf-8') as f:
        entity_data = json.load(f)
    
    # 2. 加载关系
    # 有些版本 LightRAG 文件名可能是 relations.json 或 relationships.json，这里做个容错
    if not os.path.exists(rel_path) and "relationships" in rel_path:
        # 尝试备用文件名
        alt_path = rel_path.replace("relationships", "relations")
        if os.path.exists(alt_path):
            rel_path = alt_path
            
    with open(rel_path, 'r', encoding='utf-8') as f:
        relation_data = json.load(f)
        
    # 3. 加载图结构
    G = nx.read_graphml(graph_path)
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G, entity_data, relation_data

# ================= 3. 图转换与 Leiden 分区 =================
def partition_graph(nx_graph, resolution=1.0):
    print("Converting graph and running Leiden algorithm...")
    
    edges = list(nx_graph.edges())
    nodes = list(nx_graph.nodes())
    
    if not nodes:
        return []

    node_map = {node: i for i, node in enumerate(nodes)}
    
    # 构建 igraph 对象
    # 过滤掉不在 node_map 里的边（防止 graphml 和 json 不一致导致的报错）
    valid_edges = [(node_map[u], node_map[v]) for u, v in edges if u in node_map and v in node_map]
    
    ig_graph = ig.Graph(len(nodes), valid_edges)
    
    # 运行 Leiden 算法
    partition = leidenalg.find_partition(
        ig_graph, 
        leidenalg.RBConfigurationVertexPartition, 
        resolution_parameter=resolution
    )
    
    communities = []
    for comm_indices in partition:
        comm_nodes = [nodes[i] for i in comm_indices]
        communities.append(comm_nodes)
        
    print(f"Partitioned into {len(communities)} subgraphs/communities.")
    return communities

# ================= 4. 子图内容转化 (Graph to Text) =================
def extract_subgraph_context(community_nodes, nx_graph, entity_data, relation_data):
    subgraph_content = {
        "entities": [],
        "relationships": []
    }
    
    # 提取实体
    for node_id in community_nodes:
        info = entity_data.get(node_id, {})
        desc = info.get("description", "No description available.")
        # 格式化实体信息
        subgraph_content["entities"].append(f"- {node_id} ({info.get('type', 'Unknown')}): {desc[:300]}")

    # 提取关系
    subgraph_sub = nx_graph.subgraph(community_nodes)
    for u, v, data in subgraph_sub.edges(data=True):
        rel_desc = data.get("description")
        if not rel_desc:
            possible_key = f"{u}<SEP>{v}"
            rel_info = relation_data.get(possible_key)
            if rel_info:
                rel_desc = rel_info.get("description")
        
        if rel_desc:
             subgraph_content["relationships"].append(f"- {u} -> {v}: {rel_desc[:200]}")

    return subgraph_content

# ================= 5. 构建 SLM 输入并调用 Ollama (支持长尾融合) =================
def generate_summary_payloads(communities, G, entity_data, relation_data):
    payloads = []
    
    # 1. 分离核心社区和长尾碎片
    core_communities = []
    tail_nodes = []
    
    # 阈值：节点数少于 3 的视为碎片
    MIN_SIZE = 3
    
    for nodes in communities:
        if len(nodes) >= MIN_SIZE:
            core_communities.append(nodes)
        else:
            tail_nodes.extend(nodes) # 把碎片打散，全部收集到一个大池子里
            
    print(f"Stats: Core Communities: {len(core_communities)} | Tail Nodes to merge: {len(tail_nodes)}")

    # 2. 处理核心社区 (正常流程)
    total_core = len(core_communities)
    for i, nodes in enumerate(core_communities):
        print(f"Processing Core Subgraph {i+1}/{total_core} (Nodes: {len(nodes)})...", end="", flush=True)
        
        context = extract_subgraph_context(nodes, G, entity_data, relation_data)
        
        # 核心社区 Prompt
        slm_input_text = (
            f"You are a helpful assistant.\n"
            f"Here is a specific knowledge graph subgraph:\n\n"
            f"Entities:\n" + "\n".join(context["entities"]) + "\n\n"
            f"Relationships:\n" + "\n".join(context["relationships"]) + "\n\n"
            f"Instruction:\n"
            f"Summarize the main topic and key relationships in one concise paragraph."
        )
        
        start_time = time.time()
        generated_summary = call_ollama_summary(slm_input_text)
        elapsed = time.time() - start_time
        print(f" Done ({elapsed:.2f}s)")

        payloads.append({
            "type": "core_community",
            "subgraph_id": f"core_{i}",
            "node_count": len(nodes),
            "summary": generated_summary
        })

    # 3. 处理长尾节点 (融合流程)
    # 将长尾节点打包，每包约 15 个节点，防止上下文太乱
    BATCH_SIZE = 15
    if tail_nodes:
        # 将列表切片成多个 batch
        tail_batches = [tail_nodes[i:i + BATCH_SIZE] for i in range(0, len(tail_nodes), BATCH_SIZE)]
        
        total_tail = len(tail_batches)
        print(f"Processing {total_tail} Tail Batches (Merging sparse data)...")
        
        for i, batch_nodes in enumerate(tail_batches):
            print(f"  > Merging Tail Batch {i+1}/{total_tail} (Nodes: {len(batch_nodes)})...", end="", flush=True)
            
            # 复用提取函数，虽然这些节点可能互不连接，但提取逻辑是一样的
            context = extract_subgraph_context(batch_nodes, G, entity_data, relation_data)
            
            # 长尾节点专用 Prompt (强调分类和罗列)
            slm_input_text = (
                f"You are a helpful assistant.\n"
                f"Here is a list of miscellaneous entities and fragments from a knowledge graph:\n\n"
                f"Entities:\n" + "\n".join(context["entities"]) + "\n\n"
                f"Relationships:\n" + "\n".join(context["relationships"]) + "\n\n"
                f"Instruction:\n"
                f"These entities might be disconnected. Briefly list the categories or types of information present here. "
                f"Do not try to force connections if they don't exist. Just summarize what these items are."
            )
            
            start_time = time.time()
            generated_summary = call_ollama_summary(slm_input_text)
            elapsed = time.time() - start_time
            print(f" Done ({elapsed:.2f}s)")
            
            payloads.append({
                "type": "tail_batch",
                "subgraph_id": f"tail_batch_{i}",
                "node_count": len(batch_nodes),
                "summary": generated_summary
            })

    return payloads

# ================= 主程序 =================
if __name__ == "__main__":
    try:
        # 检查 DATA_DIR 是否存在
        if not os.path.exists(DATA_DIR):
            print(f"Error: 目录不存在 - {DATA_DIR}")
            print("请修改脚本顶部的 DATA_DIR 变量指向正确的文件夹。")
            exit(1)

        # 1. Load
        G, entities, relations = load_lightrag_data(DATA_DIR)
        
        # 2. Partition
        communities = partition_graph(G, resolution=1.2)
        
        # 3. Generate Summaries (调用 Ollama)
        summaries = generate_summary_payloads(communities, G, entities, relations)
        
        # 4. Save
        # 确保输出目录存在
        output_path = os.path.join(DATA_DIR, "summary.json")
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump({
                "edge_device": "edge-a5", 
                "total_subgraphs": len(summaries),
                "model_used": MODEL_NAME,
                "subgraphs": summaries
            }, f, indent=2, ensure_ascii=False)
            
        print(f"\nSuccess! Generated {len(summaries)} summaries.")
        print(f"File saved to: {output_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")