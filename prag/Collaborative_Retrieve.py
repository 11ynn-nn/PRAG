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



async def edge_to_target_pipeline(query: str, target_device: str, noisy_vectors: list):
    
    # 4. 路由转发 (直接从传入的 noisy_vectors 列表中按索引取值)
    payload = {
        "target_device": target_device,
        "v_query_1": noisy_vectors[0],
        "v_query_2": noisy_vectors[1],
        "v_query_3": noisy_vectors[2],
        "v_query_4": noisy_vectors[3],
        "v_query_5": noisy_vectors[4],
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


async def run_architecture(query: str, target_devices: list, noisy_vectors: list, sim_threshold: float = 0.75):
    print(f"\n路由 -> {target_devices}")
    print(f"  Q: {query}")
    
    retrieved_context = ""
    # 直接遍历传进来的 target_devices，喂入传进来的 noisy_vectors
    for device in target_devices:
        # 注意：这里调用的是你之前写好的请求函数
        ctx = await edge_to_target_pipeline(query, device, noisy_vectors)
        if ctx:
            retrieved_context += f"\n{ctx}\n"
    
    # ==========================================
    # 打印检索 Context 样例
    # ==========================================
    print("\n" + "="*20 + " [检索内容预览] " + "="*20)
    print(retrieved_context.strip() if retrieved_context else "无内容")
    print("="*65 + "\n")
    # ==========================================

    if not retrieved_context:
        print(f"  [!] 注意：目标设备群 {target_devices} 未返回任何 Context")
        
    # 最好把拼好的上下文 return 出去，方便外界（比如大模型）调用
    return retrieved_context

if __name__ == "__main__":
    # ==========================================
    # 这里模拟你的主流程在外层已经算好的数据
    # ==========================================
    mock_query = "Which is a company that provides human resources management software?"
    mock_devices = ['device_0', 'device_4']
    mock_vectors = [[0.0]*768 for _ in range(10)] # 占位用的10个向量，实际跑的时候是你算好的
    
    # 直接执行这个异步函数
    result_context = asyncio.run(run_architecture(
        query=mock_query,
        target_devices=mock_devices, 
        noisy_vectors=mock_vectors, 
        sim_threshold=0.75
    ))