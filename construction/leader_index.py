import json
import os
import shutil
import chromadb
import ollama
from tqdm import tqdm

# === 配置 ===
# 你的真实摘要文件
SUMMARY_FILE = "region_b_semantic_summary.json"
# 数据库存储路径 (存到 leader-b)
LEADER_DB_PATH = os.path.join("servers_data", "leader-b", "chroma_db")

def get_embedding(text):
    """调用 Ollama 生成 768 维向量 (nomic-embed-text)"""
    try:
        res = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return res["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def build_two_layer_db():
    print(f"===========================================================")
    print(f"🚀 开始构建双层路由库 (Domain -> Summary)")
    print(f"📂 数据源: {SUMMARY_FILE}")
    print(f"📂 目标库: {LEADER_DB_PATH}")
    print(f"===========================================================")

    # 1. 读取 JSON 文件
    if not os.path.exists(SUMMARY_FILE):
        print(f"❌ 错误: 找不到文件 {SUMMARY_FILE}")
        return

    with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 初始化 ChromaDB
    if os.path.exists(LEADER_DB_PATH):
        try:
            shutil.rmtree(LEADER_DB_PATH) # 清理旧数据，保证纯净
            print("✅ 旧数据库已清理")
        except:
            pass
            
    os.makedirs(LEADER_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=LEADER_DB_PATH)
    
    # === 创建两个集合 ===
    # 1. 存放领域 (用于第一步粗排)
    domain_col = client.create_collection(name="domain_routing_vdb")
    # 2. 存放详细摘要 (用于第二步精排)
    summary_col = client.create_collection(name="summary_routing_vdb")

    # === Step A: 构建 Domain 索引 ===
    domain_list = data.get("final_domain_list", [])
    print(f"\n[1/2] 正在构建 Domain 索引 (共 {len(domain_list)} 个领域)...")
    
    d_ids, d_docs, d_metas, d_embs = [], [], [], []
    
    for domain in tqdm(domain_list, desc="Domains"):
        # 为了让向量更有语义，我们可以稍微扩充一下 Prompt
        # 例如: "Society" -> "The knowledge domain regarding Society"
        prompt_text = f"The knowledge domain of {domain}"
        
        vec = get_embedding(prompt_text)
        if not vec: continue

        d_ids.append(f"domain_{domain}")
        d_docs.append(domain)
        # Metadata 存 domain_name，方便后续 filter 使用
        d_metas.append({"domain_name": domain}) 
        d_embs.append(vec)

    if d_ids:
        domain_col.add(ids=d_ids, documents=d_docs, metadatas=d_metas, embeddings=d_embs)
    print(f"   ✅ Domain 索引构建完成。")

    # === Step B: 构建 Summary 索引 ===
    # 结构: domains -> {DomainName} -> List of Subgraphs
    domains_data = data.get("domains", {})
    
    print(f"\n[2/2] 正在构建 Summary 索引 (关联 source_edge)...")
    
    s_ids, s_docs, s_metas, s_embs = [], [], [], []
    count = 0
    
    for domain_name, subgraphs in tqdm(domains_data.items(), desc="Processing Summaries"):
        for sg in subgraphs:
            summary_text = sg.get("summary", "")
            source_edge = sg.get("source_edge", "Unknown")
            
            if not summary_text: continue
            
            vec = get_embedding(summary_text)
            if not vec: continue

            s_ids.append(f"sum_{count}")
            s_docs.append(summary_text)
            
            # 【关键】Metadata 中必须包含 'domain'，以便在第二步查询时做 filter
            s_metas.append({
                "source_edge": source_edge,
                "domain": domain_name, # <--- 关键字段
                "subgraph_id": sg.get("subgraph_id", "unknown")
            })
            s_embs.append(vec)
            count += 1
            
            # 批量写入 (每 100 条写一次)
            if len(s_ids) >= 100:
                summary_col.add(ids=s_ids, documents=s_docs, metadatas=s_metas, embeddings=s_embs)
                s_ids, s_docs, s_metas, s_embs = [], [], [], []

    # 写入剩余
    if s_ids:
        summary_col.add(ids=s_ids, documents=s_docs, metadatas=s_metas, embeddings=s_embs)

    print(f"\n🎉 双层路由库构建完成！")
    print(f"   - Domain Collection: {len(domain_list)} entries")
    print(f"   - Summary Collection: {count} entries")
    print(f"   - Path: {LEADER_DB_PATH}")

if __name__ == "__main__":
    build_two_layer_db()