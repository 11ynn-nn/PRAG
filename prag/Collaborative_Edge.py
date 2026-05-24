import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from nano_graphrag import GraphRAG
import ollama
import re

app = FastAPI(title="Multi-Device Collaborative GraphRAG Server")

class NomicOllamaEmbedding:
    def __init__(self):
        self.embedding_dim = 768  # 核心：必须明确告诉框架维度是 768
        self.max_token_size = 8192
        
    async def __call__(self, texts: list[str]) -> np.ndarray:
        embeds = []
        for t in texts:
            resp = ollama.embeddings(model='nomic-embed-text', prompt=t)
            embeds.append(resp['embedding'])
        return np.array(embeds)

# ==========================================
# 1. 预加载所有目标设备的本地图谱库
# ==========================================
print("[*] 正在初始化多设备图谱集群...")
DEVICE_DIRS = {
    "device_0": "servers_data_hotpot/device_0",
    "device_1": "servers_data_hotpot/device_1",
    "device_2": "servers_data_hotpot/device_2",
    "device_3": "servers_data_hotpot/device_3",
    "device_4": "servers_data_hotpot/device_4",
    "device_5": "servers_data_hotpot/device_5",
    "device_6": "servers_data_hotpot/device_6",
    "device_7": "servers_data_hotpot/device_7",
    "device_8": "servers_data_hotpot/device_8",
    "device_9": "servers_data_hotpot/device_9"
}

# 缓存所有设备的 RAG 实例
rag_cluster = {}
for device_name, path in DEVICE_DIRS.items():
    try:
        rag_cluster[device_name] = GraphRAG(
            working_dir=path,
            embedding_func=NomicOllamaEmbedding() 
        )
        print(f"  [✓] {device_name} 图谱加载成功")
    except Exception as e:
        print(f"  [x] {device_name} 加载失败: {e}")

# ==========================================
# 2. 接口数据模型：包含目标设备与 5 个马赛克向量
# ==========================================
class BypassRequest(BaseModel):
    target_device: str
    v_query_1: list[float]  # 对应你的修改
    v_query_2: list[float]
    v_query_3: list[float]
    v_query_4: list[float]
    v_query_5: list[float]
    top_k_broad: int = 300  # 默认值同步调大
    final_k: int = 20

# ==========================================
# 3. 核心 API：定向设备检索与图交集组装
# ==========================================
@app.post("/bypass_search")
async def bypass_search(req: BypassRequest):
    if req.target_device not in rag_cluster:
        raise HTTPException(status_code=404, detail=f"未找到目标设备 {req.target_device} 的图谱库")
    
    target_rag = rag_cluster[req.target_device]

    target_vdb = target_rag.entities_vdb
    # 获取底层的 NetworkX 图对象
    target_nx_graph = target_rag.chunk_entity_relation_graph._graph
    # ========================================================
    # 终极修复：动态劫持 Embedding 函数，实现纯向量穿透！
    # ========================================================
    async def fetch_candidates(vec):
        try:
            vec_array = np.array(vec, dtype=np.float32)
            res = None
            
            # 方案A：优先寻找底层的纯向量客户端 (通常是 _client 或 client)
            vdb_client = getattr(target_vdb, '_client', getattr(target_vdb, 'client', None))
            
            if vdb_client and hasattr(vdb_client, 'query'):
                # 底层纯向量查库 (nano-vectordb 的 query 是同步的)
                res = vdb_client.query(query=vec_array, top_k=req.top_k_broad)
            else:
                # 随便传个假文本，但把它的转换引擎换掉
                original_func = target_vdb.embedding_func
                
                async def fake_embed(texts):
                    # 劫持：忽略传入的假文本，永远返回边缘节点发来的马赛克向量！
                    return [vec_array]
                
                # 1. 偷换引擎
                target_vdb.embedding_func = fake_embed
                # 2. 用假文本骗过 Pydantic 校验
                res = await target_vdb.query(query="bypass_string", top_k=req.top_k_broad)
                # 3. 恢复原状，不留痕迹
                target_vdb.embedding_func = original_func 

            # 解析各种可能的返回格式
            if res and isinstance(res, list):
                 if isinstance(res[0], dict) and 'id' in res[0]:
                     return [r['id'] for r in res]
                 elif hasattr(res[0], 'id'):
                     return [r.id for r in res]
                 elif isinstance(res[0], dict) and 'entity_name' in res[0]:
                     return [r['entity_name'] for r in res]
            return []
        except Exception as e:
            print(f"底层向量查询异常: {e}")
            return []

    # 1. 在目标设备上进行四路粗召回
    cand_q1 = await fetch_candidates(req.v_query_1)
    cand_q2 = await fetch_candidates(req.v_query_2)
    cand_q3 = await fetch_candidates(req.v_query_3)
    cand_q4 = await fetch_candidates(req.v_query_4)
    cand_q5 = await fetch_candidates(req.v_query_5)

    # 合并所有候选池
    all_candidates = set(cand_q1 + cand_q2 + cand_q3 + cand_q4 + cand_q5)

    # 2. 图交集连通性打分 (修复黑洞效应，语义绝对优先)
    node_scores = {}
    for node in all_candidates:
        semantic_score = 0.0
        
        # A. 纯语义打分
        if node in cand_q1: semantic_score += 2.0
        if node in cand_q2: semantic_score += 2.0
        if node in cand_q3: semantic_score += 2.0
        if node in cand_q4: semantic_score += 2.0
        if node in cand_q5: semantic_score += 2.0
        
        # 拦截
        if semantic_score < 2.0:
            continue
            
        # B. 严格控制的图拓扑提权
        topo_score = 0.0
        if node in target_nx_graph:
            # 只看高质量邻居
            valid_neighbors = [
                n for n in target_nx_graph.neighbors(node) 
                if n in cand_q1 or n in cand_q2 or n in cand_q3 or n in cand_q4 or n in cand_q5
            ]
            # 邻居关系分封顶 2.0
            topo_score = min(len(valid_neighbors) * 0.5, 2.0) 
            
        node_scores[node] = semantic_score + topo_score

    # ==========================================
    # 3. 关系拓扑抽取 (保留图特征供 LLM 参考)
    # ==========================================
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    actual_k = max(req.final_k, 50) 
    core_nodes = [n for n, s in sorted_nodes[:actual_k]]
    
    relations_ctx = []
    processed_edges = set()
    
    for node in core_nodes:
        if node not in target_nx_graph: continue
        for neighbor in target_nx_graph.neighbors(node):
            edge_id = tuple(sorted([node, neighbor]))
            if edge_id not in processed_edges:
                processed_edges.add(edge_id)
                edge_data = target_nx_graph.get_edge_data(node, neighbor)
                rel_desc = edge_data.get('description', edge_data.get('relation', 'Related to')) if edge_data else 'Related to'
                
                clean_node = node.replace('"', '').strip(' |(')
                clean_neighbor = neighbor.replace('"', '').strip(' |(')
                clean_desc = rel_desc.replace('"', '').strip(' |(')
                relations_ctx.append(f'[{clean_node}]--> [{clean_neighbor}]：{clean_desc}')

    # ==========================================
    # 4. 全量提取候选池 Chunk
    # ==========================================
    target_chunk_ids = set()
    
    # 把 initial vector 扫到的所有 node 全部扒出源文档！
    for node in all_candidates:
        if node not in target_nx_graph: continue
        node_data = target_nx_graph.nodes[node]
        
        # 捞取本体 Chunk
        raw_source = node_data.get("source_id", "")
        if isinstance(raw_source, list):
            target_chunk_ids.update([str(c).strip() for c in raw_source])
        elif isinstance(raw_source, str) and raw_source:
            target_chunk_ids.update([c.strip() for c in raw_source.split(",") if c.strip()])

        # 捞取 1-hop 邻居 Chunk
        for neighbor in target_nx_graph.neighbors(node):
            n_data = target_nx_graph.nodes[neighbor]
            n_raw_source = n_data.get("source_id", "")
            if isinstance(n_raw_source, list):
                target_chunk_ids.update([str(c).strip() for c in n_raw_source])
            elif isinstance(n_raw_source, str) and n_raw_source:
                target_chunk_ids.update([c.strip() for c in n_raw_source.split(",") if c.strip()])

    # 获取真实的 Chunk 内容
    context_map = {}
    for cid in target_chunk_ids:
        if not cid or cid in context_map: continue
        try:
            cdata = None
            if hasattr(target_rag.text_chunks, 'get_by_id'):
                cdata = await target_rag.text_chunks.get_by_id(cid)
            elif hasattr(target_rag.text_chunks, 'get'):
                cdata = await target_rag.text_chunks.get(cid)
            
            if cdata:
                if isinstance(cdata, dict) and "content" in cdata:
                    context_map[cid] = cdata["content"]
                elif isinstance(cdata, str):
                    context_map[cid] = cdata
        except Exception as e:
            pass

        # ==========================================
        # 提取关系
        # ==========================================
        for neighbor in target_nx_graph.neighbors(node):
            edge_id = tuple(sorted([node, neighbor]))
            if edge_id not in processed_edges:
                processed_edges.add(edge_id)
                edge_data = target_nx_graph.get_edge_data(node, neighbor)
                rel_desc = edge_data.get('description', edge_data.get('relation', 'Related to')) if edge_data else 'Related to'
                
                # 清洗双引号，并去除边缘残留的杂质字符如 | 或 (
                clean_node = node.replace('"', '').strip(' |(')
                clean_neighbor = neighbor.replace('"', '').strip(' |(')
                clean_desc = rel_desc.replace('"', '').strip(' |(')
                
                relations_ctx.append(f'[{clean_node}]--> [{clean_neighbor}]：{clean_desc}')

    # 4. 按照新格式拼装最终文本
    final_context = "--------relations--------\n"
    final_context += "\n".join(relations_ctx)
    final_context += "\n\n--------context--------\n"
    
    # 组装时，给每个 chunk 加上明确的界限，并格式化 Title
    for chunk_content in context_map.values():
        if chunk_content:
            cleaned_chunk = re.sub(r'(?m)^Title:\s*(.*)', r'【Title: \1】：', chunk_content)
            final_context += f"{cleaned_chunk}\n"
    
    return {"status": "success", "context": final_context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)