import json
import asyncio
import numpy as np
import requests
import ollama
from openai import AsyncOpenAI
import re

# 在配置部分增加
VLLM_API_BASE = "http://localhost:8000/v1"
vllm_client = AsyncOpenAI(
    base_url=VLLM_API_BASE,
    api_key="EMPTY"
)

# 10路加噪query发送到目标设备
# ==========================================
def apply_noise(vec: list, noise_level: float) -> list:
    v_arr = np.array(vec)
    noisy_v = v_arr + np.random.normal(0, noise_level, v_arr.shape)
    norm = np.linalg.norm(noisy_v)
    return (noisy_v / norm).tolist() if norm > 0 else noisy_v.tolist()

async def edge_to_target_pipeline(query: str, target_device: str, noise_level: float = 0.3):
    
    texts_to_embed = [query]
    clean_vectors = [ollama.embeddings(model='nomic-embed-text', prompt=f"search_query: {t}")['embedding'] for t in texts_to_embed]

    # 3. 施加隐私马赛克 【引入蒙特卡洛降噪思想】
    v_query_noisy_1 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_2 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_3 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_4 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_5 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_6 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_7 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_8 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_9 = apply_noise(clean_vectors[0], noise_level)
    v_query_noisy_10 = apply_noise(clean_vectors[0], noise_level)


    # 4. 路由转发
    payload = {
        "target_device": target_device,
        "v_query_1": v_query_noisy_1,
        "v_query_2": v_query_noisy_2,
        "v_query_3": v_query_noisy_3,
        "v_query_4": v_query_noisy_4,
        "v_query_5": v_query_noisy_5,
        "v_query_6": v_query_noisy_6,
        "v_query_7": v_query_noisy_7,
        "v_query_8": v_query_noisy_8,
        "v_query_9": v_query_noisy_9,
        "v_query_10": v_query_noisy_10,
        "top_k_broad": 40,  
        "final_k": 10
    }

    try:
        response = requests.post("http://127.0.0.1:8001/bypass_search", json=payload)
        response.raise_for_status()
        ctx = response.json().get('context', "")
        if not ctx: print(f"  [!] Warning: Device {target_device} returned EMPTY context.")
        return ctx
    except Exception:
        return ""

# ==========================================
# 3. 全量测试与双重评估报告生成
# ==========================================
async def run_architecture_evaluation(test_file: str, noise_level: float = 0.3, sim_threshold: float = 0.75):
    print(f"\n=======================================================")
    print(f" 开始分布式路由协作评估 | Noise: {noise_level} | 阈值: {sim_threshold}")
    print(f"=======================================================\n")

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_queries = len(data)
    semantic_hits, exact_hits = 0, 0
    scores = []

    for i, item in enumerate(data):
        query = item['question']
        ground_truth = item.get('ground_truth', '')
        exact_answer = item.get('exact_answer', item.get('answer', '')).lower() 
        target_device = item['device'] # 从 JSON 读取目标靶点

        print(f"[{i+1}/{total_queries}] 路由 -> [{target_device}]")
        print(f"  Q: {query}")
        
        # 核心动作：向目标设备发送 5 个加噪向量
        retrieved_context = await edge_to_target_pipeline(query, target_device, noise_level)
        
        # ==========================================
        # 【新增】打印第一个问题的检索 Context 样例
        # ==========================================
        if i == 0:
            print("\n" + "="*20 + " [第一个问题检索内容预览] " + "="*20)
            print(retrieved_context)
            print("="*65 + "\n")
        # ==========================================

        if not retrieved_context:
            print(f"  [!] 注意：设备 {target_device} 未返回任何 Context")

        # 评估
        is_exact_hit = exact_answer in retrieved_context.lower() if exact_answer else False
        if is_exact_hit: exact_hits += 1

        # 判定结论输出
        if is_exact_hit:
            print(f" 精确命中: True")
            semantic_hits += 1 
        else:
            print(f" [x] 召回失败")
        print("-" * 55)
    
    # --- 打印双重评估报告 ---
    print(f"\n==================== 最终评估报告 ====================")
    print(f"测试数据总量: {total_queries} (全路由协作模式)")
    print(f"系统注入噪声: {noise_level}")
    print(f"------------------------------------------------------")
    print(f"[指标 2] 精确词汇命中率 (Exact): {exact_hits}/{total_queries} ({(exact_hits/total_queries):.1%})")
    print(f"======================================================\n")

if __name__ == "__main__":
    asyncio.run(run_architecture_evaluation('test.json', noise_level=0.3, sim_threshold=0.75))