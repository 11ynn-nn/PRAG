import json
import requests
import numpy as np
import math
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------------- 配置区 ----------------
CLOUD_DB_PATH = "servers_data_hotpot/leader_1" 
COLLECTION_NAME = "region_index" 
TEST_FILE = "test.json"
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
SIM_THRESHOLD = 0.54 # 召回摘要的最低平均相似度阈值

# 领域名称映射表
DOMAIN_MAPPING = {
    "CS": "computer science",
    "Sports": "sports",
    "Cooking": "cooking",
    "Medical": "medical"
}

# ---------------- 核心功能函数 ----------------
def get_ollama_embedding(text: str) -> list:
    """调用本地 Ollama API 获取文本的 Embedding 向量"""
    try:
        payload = {
            "model": EMBED_MODEL,
            "prompt": text
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("embedding", [])
    except Exception as e:
        print(f"Error fetching embedding for {text}: {e}")
        return [0.0] * 768  # 失败时返回 768 维全零向量打底
        
def cosine_similarity(v1: list, v2: list) -> float:
    """计算两个向量的余弦相似度，用于第一阶段 Domain 匹配"""
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(y * y for y in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

print(f"连接本地模拟云端: {CLOUD_DB_PATH} ...")
client = QdrantClient(path=CLOUD_DB_PATH)

# ================= 架构升级：构建第一阶段的 Domain 路由表 =================
print("初始化 Domain 宏观路由表...")
DOMAIN_VECTORS = {
    domain: get_ollama_embedding(domain) for domain in set(DOMAIN_MAPPING.values())
}
# ====================================================================

# ---------------- 绝对隐私的两阶段路由检索函数 ----------------
def route_and_retrieve(question: str, noise_level: float = 0.3, noisy_vectors: list = None):
    
    if noisy_vectors is None:
        noisy_vectors = []
    
    # ====== 阶段 1: 纯 Domain 级别匹配 ======
    domain_scores = Counter()
    for n_vec in noisy_vectors:
        for domain_name, d_vec in DOMAIN_VECTORS.items():
            sim = cosine_similarity(n_vec, d_vec)
            domain_scores[domain_name] += sim
            
    top_domains_with_scores = domain_scores.most_common(3)
    top_domain_names = [item[0] for item in top_domains_with_scores]
    
    # ====== 阶段 2: 领域内加噪多路召回 + 平均分提取 ======
    domain_summaries = {}
    device_max_scores = {} 
    
    for domain in top_domain_names:
        unique_hits = {} # 结构: point_id -> {"device": dev_id, "text": text, "scores": []}
        
        # 拿着这 10 个加噪向量，去同一个 Domain 里狂轰乱炸 10 次
        for n_vec in noisy_vectors:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=n_vec, # <--- 【核心】这里传的不再是 clean_vector，而是加噪向量！
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="domain",
                            match=MatchValue(value=domain)
                        )
                    ]
                ),
                limit=30, 
                with_payload=True
            )
            
            # 记录每一次被捞上来的摘要
            for hit in response.points:
                if hit.id not in unique_hits:
                    unique_hits[hit.id] = {
                        "device": hit.payload.get('device_id', 'unknown'),
                        "text": hit.payload.get('summary_text', ''),
                        "scores": []
                    }
                unique_hits[hit.id]["scores"].append(hit.score)
        
        # 统计平均分并卡阈值
        summaries = []
        for point_id, data in unique_hits.items():
            # 计算该摘要被多次召回后的平均得分
            avg_score = sum(data["scores"]) / len(data["scores"])
            
            # 使用平均分来挑战 0.60 的阈值
            if avg_score >= SIM_THRESHOLD:
                dev_id = data["device"]
                hit_count = len(data["scores"])
                
                summaries.append({
                    "device": dev_id,
                    "text": data["text"],
                    "score": avg_score,
                    "hit_count": hit_count # 记录被击中次数，用于终端展示
                })
                
                # 记录该设备的最高平均得分
                if dev_id not in device_max_scores or avg_score > device_max_scores[dev_id]:
                    device_max_scores[dev_id] = avg_score
        
        # 按平均得分降序排列该领域的摘要
        summaries.sort(key=lambda x: x["score"], reverse=True)
        domain_summaries[domain] = summaries
        
    # 根据设备的最高平均得分进行降序排列，只要合格的全部下发
    routed_devices = [dev for dev, score in sorted(device_max_scores.items(), key=lambda x: x[1], reverse=True)]
        
    return top_domains_with_scores, domain_summaries, routed_devices

# ---------------- 执行测试集 ----------------
if __name__ == "__main__":
    try:
        print(f"\n加载测试数据: {TEST_FILE}")
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        correct_domain_count = 0
        correct_device_count = 0
        total_count = len(test_data)
        
        print("-" * 80)
        for i, item in enumerate(test_data):
            question = item.get("question", "")
            raw_project = item.get("project", "")
            expected_device = item.get("device", "")
            expected_domain = DOMAIN_MAPPING.get(raw_project, raw_project.lower())
            
            # 呼叫绝对隐私的两阶段路由机制
            top_domains_with_scores, domain_summaries, routed_devices = route_and_retrieve(
                question=question, 
                noise_level=0.3 
            )
            
            top_domain_names = [item[0] for item in top_domains_with_scores]
            
            domain_hit = expected_domain in top_domain_names
            device_hit = expected_device in routed_devices
            
            if domain_hit: correct_domain_count += 1
            if device_hit: correct_device_count += 1
                
            if domain_hit and device_hit:
                status = "✅ 领域与设备双杀命中"
            elif domain_hit:
                status = "⚠️ 仅领域命中 (设备摘要平均分未达到 0.60)"
            else:
                status = "❌ 完全脱靶"
                
            print(f"\n[{i+1}/{total_count}] {status}")
            print(f"问题: {question}")
            print(f"期望目标 -> Domain: {expected_domain} | Device: {expected_device}")
            
            formatted_domains = [f"'{d}': {s:.3f}" for d, s in top_domains_with_scores]
            print(f"宏观路由 -> Top-2 Domain: [{', '.join(formatted_domains)}]")
            
            if routed_devices:
                print(f"微观路由 -> 最终确定的目标设备 (平均得分 >= {SIM_THRESHOLD}): {routed_devices}")
            else:
                print(f"微观路由 -> 最终确定的目标设备: [空] (无摘要平均分 >= {SIM_THRESHOLD})")
            print("-" * 40) 
            
            print("=" * 80)
            
        print(f"\n🎉 测试完成!")
        print(f"宏观 Domain 路由准确率: {correct_domain_count}/{total_count} ({(correct_domain_count/total_count)*100:.2f}%)")
        print(f"微观 Device 路由准确率: {correct_device_count}/{total_count} ({(correct_device_count/total_count)*100:.2f}%)")
    
    except Exception as e:
        print(f"\n❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n关闭数据库连接...")
        client.close()