import json
import uuid
import os
import requests
import math
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------- 配置区 ----------------
CLOUD_DB_PATH = "servers_data_hotpot/cloud" 
COLLECTION_NAME = "region_index"
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# 定义你的四大领域分类
DOMAIN_CATEGORIES = ["computer science", "cooking", "sports", "medical"]

# ---------------- 工具函数 ----------------
def get_ollama_embedding(text: str) -> list:
    """调用本地 Ollama 接口获取向量"""
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except Exception as e:
        print(f"Ollama 请求异常: {e}")
    return []

def cosine_similarity(v1: list, v2: list) -> float:
    """计算两个向量的余弦相似度，用于智能领域分类"""
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(y * y for y in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# ---------------- 初始化阶段 ----------------
print(f"探测 Ollama 模型 [{EMBED_MODEL}] 的维度...")
test_vector = get_ollama_embedding("Hello world")
if not test_vector:
    raise Exception("Ollama 接口调用失败，请检查 Ollama 是否启动且已拉取模型")
VECTOR_SIZE = len(test_vector)
print(f"成功！向量维度为: {VECTOR_SIZE}")

# 预先计算 4 个 Domain 的基准向量，用于无监督的快速分类
print("正在计算四大领域的基准语义向量，准备开启智能路由匹配...")
DOMAIN_VECTORS = {domain: get_ollama_embedding(domain) for domain in DOMAIN_CATEGORIES}

print(f"初始化本地云端中心: {CLOUD_DB_PATH}")
client = QdrantClient(path=CLOUD_DB_PATH)

if not client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"集合 {COLLECTION_NAME} 不存在，正在创建...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="domain",
        field_schema="keyword"
    )
    print("集合及 Payload 索引创建完毕。")
else:
    print(f"集合 {COLLECTION_NAME} 已存在。")

# ---------------- 核心提取与存储逻辑 ----------------
def upload_device_summaries(device_id: str, filepath: str):
    print(f"\n--- 开始处理 {device_id} ---")
    
    if not os.path.exists(filepath):
        print(f"⚠️ 跳过: 找不到文件 {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = []
    
    # 兼容 GraphRAG / LightRAG 生成的特有字典格式
    report_items = data.items() if isinstance(data, dict) else enumerate(data)
    
    for original_id, report_data in report_items:
        summary = ""
        
        # 1. 深入剥析 JSON，提取躲在最深处的 summary
        if isinstance(report_data, dict):
            report_json = report_data.get('report_json', {})
            # 有时 report_json 会被转义成字符串格式的 JSON，防止报错，帮它反序列化
            if isinstance(report_json, str):
                try:
                    report_json = json.loads(report_json)
                except json.JSONDecodeError:
                    report_json = {}
                    
            if isinstance(report_json, dict):
                summary = report_json.get('summary', '')
                
            # 兜底：如果最外层有的话也拿一下
            if not summary:
                summary = report_data.get('summary', report_data.get('text', ''))
                
        if not summary:
            continue
            
        # 2. 计算 Summary 的 Embedding
        vector = get_ollama_embedding(summary)
        if not vector:
            continue
            
        # 3. 动态智能分类 Domain (核心亮点：用向量相似度极速判断)
        best_domain = "unknown"
        highest_sim = -1.0
        for domain, d_vec in DOMAIN_VECTORS.items():
            sim = cosine_similarity(vector, d_vec)
            if sim > highest_sim:
                highest_sim = sim
                best_domain = domain
                
        # 4. 生成唯一 ID 并组装进 Payload
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{device_id}_{original_id}"))
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "domain": best_domain,
                "device_id": device_id,
                "summary_text": summary
            }
        ))
    
    # 批量上传
    if points:
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points
        )
        # 打印一下智能分类的结果，让你直观看到算法干了什么
        sample_domains = [p.payload['domain'] for p in points[:3]]
        print(f"✅ 成功聚合 {len(points)} 条摘要到云端！ (前3条摘要被智能分配到了: {sample_domains})")
    else:
        print("⚠️ 未找到有效的摘要数据。")

# ---------------- 执行 ----------------
if __name__ == "__main__":
    devices = ["device_0", "device_1", "device_2", "device_3", "device_4","device_5", "device_6", "device_7", "device_8", "device_9"]
    
    for device_id in devices:
        filepath = f"servers_data_hotpot/{device_id}/kv_store_community_reports.json"
        upload_device_summaries(device_id, filepath)
        
    print("\n🎉 所有设备的摘要聚合完毕，全局路由索引已建立！")
    
    # 手动安全关闭客户端，彻底解决 Python 退出时 sys.meta_path is None 的红字报错！
    client.close()