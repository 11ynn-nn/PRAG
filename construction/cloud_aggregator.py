import chromadb
from chromadb.utils import embedding_functions
import json
import os

# ================= 配置 =================
INPUT_FILES = [
    "servers_data/edge-b1/semantic_subgraphs.json",
    "servers_data/edge-b2/semantic_subgraphs.json",
    "servers_data/edge-b3/semantic_subgraphs.json",
    "servers_data/edge-b4/semantic_subgraphs.json",
    "servers_data/edge-b5/semantic_subgraphs.json"
    # 可以在这里添加其他节点的文件
]
DB_PATH = "servers_data/leader-b/chroma_db"

EMBED_FUNC = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

def ingest_data():
    print("=== Cloud Ingestion: Tagged Virtual Subgraphs ===")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 清理旧集合 (演示用)
    try: client.delete_collection("edge_summaries") 
    except: pass
    
    collection = client.create_collection(
        name="edge_summaries", 
        embedding_function=EMBED_FUNC,
        metadata={"hnsw:space": "cosine"}
    )

    total_vectors = 0
    
    for fpath in INPUT_FILES:
        if not os.path.exists(fpath): continue
        
        print(f"📂 Reading {fpath}...")
        with open(fpath, 'r', encoding='utf-8') as f:
            items = json.load(f)
            
        ids = []
        docs = []
        metas = []
        
        for item in items:
            # 构造唯一 ID
            unique_id = f"{item['edge_id']}_sub_{item['subgraph_index']}"
            
            # 构造内容：为了增强语义，可以把 Domain 加在文本前面
            doc_content = f"Domain: {item['domain_label']}\nContent: {item['summary']}"
            
            ids.append(unique_id)
            docs.append(doc_content)
            
            # 关键：Metadata 记录语义分区
            metas.append({
                "edge_id": item['edge_id'],
                "domain": item['domain_label'], # <--- 以后检索就靠它过滤
                "virtual_group": item['virtual_group_id']
            })
            
        # 批量写入
        if ids:
            batch = 20
            for i in range(0, len(ids), batch):
                collection.add(
                    ids=ids[i:i+batch], 
                    documents=docs[i:i+batch], 
                    metadatas=metas[i:i+batch]
                )
            total_vectors += len(ids)
            
    print(f"✅ Ingested {total_vectors} vectors.")
    print("   Semantic Tags are saved in Metadata.")

if __name__ == "__main__":
    ingest_data()