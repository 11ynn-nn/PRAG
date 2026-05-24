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

async def edge_to_target_pipeline(target_device: str, noisy_vectors: list):
    
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
        # 修正：使用 asyncio.to_thread 将同步的 requests 包裹为异步非阻塞执行，无需额外安装 httpx
        response = await asyncio.to_thread(
            requests.post, 
            "http://127.0.0.1:8001/bypass_search", 
            json=payload
        )
        response.raise_for_status()
        ctx = response.json().get('context', "")
        if not ctx: 
            print(f"  [!] Warning: Device {target_device} returned EMPTY context.")
        return ctx
    except Exception as e:
        # 建议保留打印异常，方便调试设备接口不通的问题
        # print(f"  [!] Error connecting to {target_device}: {e}")
        return ""


async def run_architecture(target_devices: list, noisy_vectors: list, sim_threshold: float = 0.75):
    print(f"\n路由 -> {target_devices}")
    
    retrieved_context = ""
    # 遍历传进来的 target_devices，喂入传进来的 noisy_vectors
    for device in target_devices:
        ctx = await edge_to_target_pipeline(device, noisy_vectors)
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
        
    return retrieved_context

if __name__ == "__main__":
    # ==========================================
    # 模拟主流程在外层算好的数据
    # ==========================================
    mock_devices = ['device_0', 'device_4']
    mock_vectors = [[0.0]*768 for _ in range(10)] # 占位用的10个向量
    
    # 修正：删除了这里多余的 query 参数，完全对齐 run_architecture 的定义
    result_context = asyncio.run(run_architecture(
        target_devices=mock_devices, 
        noisy_vectors=mock_vectors, 
        sim_threshold=0.75
    ))