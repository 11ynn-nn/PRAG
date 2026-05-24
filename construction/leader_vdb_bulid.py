import json
import os
import time
import chromadb
import ollama
from tqdm import tqdm

# ================= 配置 =================
LEADER_ID = "cloud"
JSON_FILE = "cloud_semantic_summary.json"
DB_PATH = os.path.join("servers_data", LEADER_ID, "chroma_db")
EMBED_MODEL = "nomic-embed-text"

# 十大超级领域的描述词（用于 Level 1 路由库）
SUPER_DOMAIN_DESCRIPTIONS = {
    "Entertainment_Media": "Video games, movies, music, TV shows, celebrities, news, journalism, social media, esports, arts, performance.",
    "Tech_Engineering": "Technology, computers, AI, software, hardware, internet, networks, automotive, cars, cryptography, security, logistics, supply chain.",
    "Business_Economy": "Finance, banking, stock market, investment, economy, trade, GDP, entrepreneurship, startups, corporate business.",
    "Physical_Sciences": "Natural science, physics, chemistry, astronomy, space, geology, earth science, hydrology, water, scientific research.",
    "Life_Sciences": "Biology, medicine, health, physiology, genetics, ecology, environment, nature, agriculture, fossils, paleontology, taxonomy.",
    "Formal_Sciences": "Mathematics, algebra, geometry, logic, statistics, data analysis, probability, topology, abstract structures.",
    "Society_History": "History, civilization, past events, sociology, culture, traditions, psychology, human behavior, geography, education, religion.",
    "Politics_Law": "Government, politics, diplomacy, law, justice, legal systems, elections, voting, warfare, military, international relations.",
    "Arts_Humanities": "Literature, books, writing, authors, philosophy, ethics, grammar, language, linguistics, visual arts.",
    "Sports_Competition": "Sports, athletics, olympics, football, basketball, MMA, fighting, hunting, outdoor activities, competition."
}

def get_embedding(text):
    """生成 nomic-embed-text 向量"""
    for attempt in range(3):
        try:
            # 💡 Nomic 模型推荐使用 search_document: 前缀来处理待检索文档
            res = ollama.embeddings(model=EMBED_MODEL, prompt=f"search_document: {text}")
            return res["embedding"]
        except Exception as e:
            time.sleep(1)
    return None

def main():
    print(f"📦 加载重构后的数据: {JSON_FILE}")
    if not os.path.exists(JSON_FILE):
        print(f"❌ 找不到文件 {JSON_FILE}，请先运行 reconstruct_json.py")
        return

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化 ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)

    # --- 1. 构建 Domain 路由库 (Level 1) ---
    print("\n🚀 正在构建 Level 1: Super Domain VDB...")
    try: client.delete_collection("domain_routing_vdb")
    except: pass
    dom_coll = client.create_collection("domain_routing_vdb")

    for name, desc in SUPER_DOMAIN_DESCRIPTIONS.items():
        # 这里的文本设计直接影响第一层路由的准确率
        input_text = f"Cluster: {name}. Keywords: {desc}"
        vec = get_embedding(input_text)
        if vec:
            dom_coll.add(
                ids=[f"super_{name}"],
                embeddings=[vec],
                metadatas=[{"domain_name": name}],
                documents=[input_text]
            )
    print(f"✅ Level 1 完成。")

    # --- 2. 构建 Summary 路由库 (Level 2) ---
    print("\n🚀 正在构建 Level 2: Merged Summary VDB...")
    try: client.delete_collection("summary_routing_vdb")
    except: pass
    sum_coll = client.create_collection("summary_routing_vdb")

    domains_data = data.get("domains", {})
    total_summaries = sum(len(v) for v in domains_data.values())
    pbar = tqdm(total=total_summaries, desc="Embedding Summaries")

    for super_name, items in domains_data.items():
        batch_ids, batch_vecs, batch_metas, batch_docs = [], [], [], []
        
        for idx, item in enumerate(items):
            content = item.get("summary", "")
            if not content:
                pbar.update(1)
                continue

            # 嵌入文本只包含超级领域和内容，保持纯净
            embed_input = f"Cluster: {super_name}. Content: {content}"
            vec = get_embedding(embed_input)
            
            if vec:
                batch_ids.append(f"sum_{super_name}_{idx}")
                batch_vecs.append(vec)
                batch_metas.append({
                    "domain": super_name,
                    "source_edge": item.get("source_edge"),
                    "subgraph_id": item.get("subgraph_id")
                })
                batch_docs.append(content)

            pbar.update(1)

            # 分批写入防止内存压力
            if len(batch_ids) >= 50:
                sum_coll.add(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_metas, documents=batch_docs)
                batch_ids, batch_vecs, batch_metas, batch_docs = [], [], [], []

        # 写入剩余部分
        if batch_ids:
            sum_coll.add(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_metas, documents=batch_docs)

    pbar.close()
    print(f"\n✨ 全部重构完成！数据库保存在: {DB_PATH}")

if __name__ == "__main__":
    main()