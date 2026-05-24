import json
import os
import asyncio
import numpy as np
import time
import ollama
import traceback
from openai import AsyncOpenAI
# 修改点 1：更换为 nano-graphrag 导入，移除 LightRAG 依赖
from nano_graphrag import GraphRAG, QueryParam
import nest_asyncio
nest_asyncio.apply()

# ================= 配置部分 =================
DATASET_PATH = "knowledge_base_hotpot/device_0/questions.json" 
WORKING_DIR = "servers_data_hotpot/device_0"   
OUTPUT_FILE = "local_test_0.json"  
NODE_ID = "device_0"

EMBEDDING_MODEL = "nomic-embed-text" 
EMBEDDING_DIM = 768

VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "qwen/Qwen2.5-7B-Instruct"

vllm_client = AsyncOpenAI(
    base_url=VLLM_API_BASE,
    api_key="EMPTY"
)

# ================= 函数部分 =================

class EmbeddingWrapper:
    def __init__(self, func, dim):
        self.func = func
        self.embedding_dim = dim
    async def __call__(self, texts):
        return await self.func(texts)

async def _ollama_embedding_func(texts: list[str]) -> np.ndarray:
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        return np.array(response["embeddings"])
    except:
        return np.zeros((len(texts), EMBEDDING_DIM))

ollama_embedding_obj = EmbeddingWrapper(_ollama_embedding_func, EMBEDDING_DIM)

async def vllm_model_complete(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = await vllm_client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 800)
        )
        # 修改点 2：确保返回字符串，防止 NoneType 报错
        content = response.choices[0].message.content
        return content if content is not None else ""
    except Exception as e:
        print(f"vLLM Generation Error: {e}")
        return ""

# ================= 核心类 =================

class Edge_Retrieve:
    def __init__(self, node_id, working_dir):
        self.node_id = node_id
        self.working_dir = working_dir
        self.is_ready = False
        self.local_decision = None
        self.vllm_client = vllm_client

    async def initialize(self):
        print(f"[{self.node_id}] Initializing GraphRAG engine...")
        if not os.path.exists(self.working_dir): os.makedirs(self.working_dir)

        self.rag = GraphRAG(
            working_dir=self.working_dir,
            best_model_func=vllm_model_complete,
            cheap_model_func=vllm_model_complete,
            embedding_func=ollama_embedding_obj  # 使用包装后的对象
        )
        self.is_ready = True
        print(f"[{self.node_id}] System Ready.")

    # 修改点 4：将方法名从 query 改为 Local_Inference，以匹配 main 函数调用
    async def Local_Inference(self, query: str):
        """执行本地 RAG 检索：自动探测类型，彻底解决 await 报错"""
        try:
            # 1. 发起请求
            # 注意：这里我们只 await 函数本身的执行
            result = self.rag.query(query, param=QueryParam(mode="local"))
            
            # 2. 动态判断返回结果的类型
            if asyncio.iscoroutine(result):
                # 如果返回的是协程对象，我们才 await 它
                prediction = await result
            else:
                # 如果它直接返回了字符串（同步执行结果），直接使用
                prediction = result
            
            # 3. 最后的双重保险：确保结果是字符串
            return str(prediction) if prediction is not None else ""
            
        except Exception as e:
            # 如果这里报错，说明是 RAG 内部逻辑问题，不再是语法问题
            print(f"RAG Query Error: {e}")
            return ""

    async def _evaluate_confidence(self, question: str, response: str) -> str:
        """评估生成结果的置信度"""
        # 修改点 5：增加基础判空，防止 response 为 None 时 lower() 报错
        if not response:
            return 'collaborative'

        judge_prompt = (
            f"Task: Check if the AI response successfully answers the query using the provided context.\n\n"
            f"Query: \"{question}\"\n"
            f"Response: \"{response}\"\n\n"
            f"Verdict (CONFIDENT / UNCERTAIN):"
        )

        try:
            res = await self.vllm_client.chat.completions.create(
                model=VLLM_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": "You are a strict judge. Output only: CONFIDENT or UNCERTAIN."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.0,  
                max_tokens=16     
            )
            judgment = res.choices[0].message.content.strip().upper()
            return 'collaborative' if "UNCERTAIN" in judgment else 'Local'
            
        except Exception as e:
            print(f"Judge Evaluation Error: {e}")
            return 'collaborative'

# ================= 主流程 =================

async def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        if isinstance(raw_data, dict) and "test_cases" in raw_data:
            questions_data = raw_data["test_cases"]
        elif isinstance(raw_data, list):
            questions_data = raw_data
        else:
            print("Error: 未能找到测试数据。")
            return
    except Exception as e:
        print(f"Error loading dataset: {e}"); return

    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except: pass
    
    processed_ids = set([str(item.get("question_id")) for item in results])

    edge = Edge_Retrieve(node_id=NODE_ID, working_dir=WORKING_DIR)
    await edge.initialize()

    total = len(questions_data)
    for idx, item in enumerate(questions_data):
        q_id = str(item.get("id", idx))
        query = item.get("question", "")
        
        if not query or q_id in processed_ids:
            continue

        print(f"\n[{idx+1}/{total}] Processing ID {q_id}...")
        total_start_time = time.time() # 记录总流程开始时间

        try:
            # ================= 1. 计算本地推理时间 =================
            inference_start_time = time.time()
            answer = await edge.Local_Inference(query)
            inference_time = time.time() - inference_start_time
            # =======================================================

            print(f"  -> Evaluating confidence...")
            
            # ================= 2. 计算门控决策时间 =================
            gate_start_time = time.time()
            decision = await edge._evaluate_confidence(query, answer)
            gate_time = time.time() - gate_start_time
            # =======================================================
            
            # 将计算出的时间加入字典记录
            record = {
                "question_id": q_id,
                "question": query,
                "answer": answer if answer else "No answer.",
                "gate_decision": decision,
                "local_inference_time": round(inference_time, 4), # 保留4位小数
                "gate_time": round(gate_time, 4)                  # 保留4位小数
            }
            results.append(record)

            # 实时写入文件
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
            total_time = time.time() - total_start_time
            # 终端输出打印各个阶段耗时，方便你在控制台监控
            print(f"  -> Done! Decision: [{decision}] | Inference: {inference_time:.2f}s | Gate: {gate_time:.2f}s | Total: {total_time:.2f}s")

        except Exception as e:
            print(f"  -> Error on {q_id}: {e}")
            continue

    print(f"\nSaved {len(results)} records to {OUTPUT_FILE}.")

if __name__ == "__main__":
    asyncio.run(main())