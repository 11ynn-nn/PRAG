import json
import os
import asyncio
import numpy as np
import time
import ollama
import traceback
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
import nest_asyncio
nest_asyncio.apply()

# ================= 基础配置 =================
BASE_DATASET_DIR = "knowledge_base_hotpot"
BASE_WORKING_DIR = "servers_data_hotpot"

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
        self.vllm_client = vllm_client

    async def initialize(self):
        print(f"[{self.node_id}] Initializing GraphRAG engine...")
        if not os.path.exists(self.working_dir): os.makedirs(self.working_dir)

        self.rag = GraphRAG(
            working_dir=self.working_dir,
            best_model_func=vllm_model_complete,
            cheap_model_func=vllm_model_complete,
            embedding_func=ollama_embedding_obj
        )
        self.is_ready = True
        print(f"[{self.node_id}] System Ready.")

    async def Local_Inference(self, query: str):
        try:
            result = self.rag.query(query, param=QueryParam(mode="local"))
            if asyncio.iscoroutine(result):
                prediction = await result
            else:
                prediction = result
            return str(prediction) if prediction is not None else ""
        except Exception as e:
            print(f"RAG Query Error: {e}")
            return ""

    async def _evaluate_confidence(self, question: str, response: str) -> tuple[str, str]:
        if not response:
            return 'collaborative', "Empty response."

        # ================= 极简字面派 System Prompt =================
        system_prompt = """You are a simple, literal-minded evaluator for an AI Q&A system. 
Your ONLY job is to look for explicit confessions of failure in the AI's response. Do NOT overthink or fact-check the AI.

CRITICAL RULES:
1. EXPLICIT LACK OF CONTEXT = UNCERTAIN: If the AI EXPLICITLY uses phrases like "not provided in the text", "not mentioned in the tables", "cannot determine", or "no information", you must output UNCERTAIN.
2. EXPLICIT EXTERNAL KNOWLEDGE = UNCERTAIN: If the AI EXPLICITLY states it is using "external knowledge", "general knowledge", or "outside sources", you must output UNCERTAIN.
3. THE DEFAULT RULE = CONFIDENT: If the AI provides a smooth answer and does NOT explicitly use the negative phrases from Rule 1 or 2, you MUST output CONFIDENT.

STRICT WARNINGS FOR THE JUDGE:
- NEVER penalize the AI for using dates (e.g., 2012, 2004), facts, or making logical inferences. Assume ALL data in the response comes from the local context unless the AI explicitly admits otherwise.
- If the AI sounds confident and gives an answer, your verdict MUST be CONFIDENT.

OUTPUT FORMAT (Must be exactly like this):
REASONING: [1 brief sentence explaining which Rule (1, 2, or 3) applies]
VERDICT: [CONFIDENT or UNCERTAIN]"""

        user_prompt = f'Query: "{question}"\nResponse: "{response}"'

        try:
            res = await self.vllm_client.chat.completions.create(
                model=VLLM_MODEL_NAME, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  
                max_tokens=256  
            )
            raw_output = res.choices[0].message.content.strip()
            
            reasoning_text = ""
            judgment = "UNCERTAIN"
            
            if "VERDICT:" in raw_output.upper():
                parts = raw_output.upper().split("VERDICT:")
                judgment = parts[-1].strip()
                
            if "REASONING:" in raw_output.upper():
                reasoning_split = raw_output.split("VERDICT:", 1)[0]
                reasoning_text = reasoning_split.replace("REASONING:", "", 1).replace("REASONING:", "", 1).strip()
            else:
                reasoning_text = raw_output.strip()

            if "CONFIDENT" in judgment and "UNCERTAIN" not in judgment:
                return 'Local', reasoning_text
            else:
                return 'collaborative', reasoning_text
            
        except Exception as e:
            print(f"Judge Evaluation Error: {e}")
            return 'collaborative', f"Error: {e}"

# ================= 单设备处理逻辑 =================
async def process_single_device(device_index: int):
    node_id = f"device_{device_index}"
    dataset_path = f"{BASE_DATASET_DIR}/{node_id}/questions.json"
    working_dir = f"{BASE_WORKING_DIR}/{node_id}"
    output_file = f"local_test_{device_index}.json"

    print(f"\n{'='*50}")
    print(f"🚀 [START] Processing {node_id}")
    print(f"{'='*50}")

    if not os.path.exists(dataset_path):
        print(f"⚠️ Warning: Dataset for {node_id} not found at {dataset_path}. Skipping.")
        return 0, 0

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict) and "test_cases" in raw_data:
            questions_data = raw_data["test_cases"]
        elif isinstance(raw_data, list):
            questions_data = raw_data
        else:
            print(f"Error: Format not recognized for {node_id}.")
            return 0, 0
    except Exception as e:
        print(f"Error loading dataset for {node_id}: {e}"); return 0, 0

    results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except: pass
    
    processed_ids = set([str(item.get("question_id")) for item in results])

    edge = Edge_Retrieve(node_id=node_id, working_dir=working_dir)
    await edge.initialize()

    total = len(questions_data)
    for idx, item in enumerate(questions_data):
        q_id = str(item.get("id", idx))
        query = item.get("question", "")
        
        correct_answer = item.get("answer", "") 
        expected_route = item.get("route_type", "")
        
        if not query or q_id in processed_ids:
            continue

        print(f"\n[{node_id}] [{idx+1}/{total}] Processing ID {q_id}...")
        total_start_time = time.time()

        try:
            inference_start_time = time.time()
            answer = await edge.Local_Inference(query)
            inference_time = time.time() - inference_start_time

            print(f"  -> Evaluating confidence...")
            
            gate_start_time = time.time()
            decision, gate_reasoning = await edge._evaluate_confidence(query, answer)
            gate_time = time.time() - gate_start_time
            
            # --- 归一化逻辑，防止 collaborative-1 这种标签误判 ---
            gate_decision_norm = decision.strip().lower()
            expected_route_norm = expected_route.strip().lower()
            if expected_route_norm.startswith("collaborative"):
                expected_route_norm = "collaborative"
            
            is_route_match = (gate_decision_norm == expected_route_norm)
            
            record = {
                "question_id": q_id,
                "question": query,
                "correct_answer": correct_answer,             
                "answer": answer if answer else "No answer.",
                "route_type": expected_route,                 
                "gate_decision": decision,
                "is_route_match": is_route_match,
                "gate_reasoning": gate_reasoning,             
                "local_inference_time": round(inference_time, 4),
                "gate_time": round(gate_time, 4)
            }
            results.append(record)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
                
            total_time = time.time() - total_start_time
            
            match_str = "✅ MATCH" if is_route_match else "❌ MISMATCH"
            print(f"  -> Done! Gate: [{decision}] vs True: [{expected_route}] ({match_str})")
            print(f"  -> Time | Inference: {inference_time:.2f}s | Gate: {gate_time:.2f}s | Total: {total_time:.2f}s")

        except Exception as e:
            print(f"  -> Error on {q_id}: {e}")
            continue

    print(f"\n✅ Finished {node_id}. Saved {len(results)} records to {output_file}.")

    # ================= 新增：单设备计算统计结果 =================
    total_count = len(results)
    correct_count = 0
    error_ids = []
    
    for item in results:
        g_dec = item.get("gate_decision", "").strip().lower()
        e_route = item.get("route_type", "").strip().lower()
        if e_route.startswith("collaborative"):
            e_route = "collaborative"
            
        if g_dec == e_route:
            correct_count += 1
        else:
            error_ids.append((item.get("question_id"), item.get("gate_decision"), item.get("route_type")))
            
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\n📊 [{node_id} 统计报告] 总计: {total_count} | 正确: {correct_count} | 准确率: {accuracy:.2f}%")
        if error_ids:
            print(f"❌ 本节点判错记录 ({len(error_ids)} 个):")
            for err in error_ids:
                print(f"   ID: {err[0]:<25} | 误判为: {err[1]:<10} | 实际应为: {err[2]}")
        else:
            print("🎉 完美！全部正确！")
            
    return total_count, correct_count

# ================= 主流程 =================
async def main():
    total_all_questions = 0
    total_all_correct = 0
    
    # 遍历 device_0 到 device_9 
    for i in range(10):
        # 接收跑完单设备的统计数据
        res = await process_single_device(i)
        if res:
            t_count, c_count = res
            total_all_questions += t_count
            total_all_correct += c_count
    
    # 打印最终全局汇总信息
    print(f"\n\n{'*'*20} 🏆 全局终极统计 {'*'*20}")
    if total_all_questions > 0:
        overall_acc = (total_all_correct / total_all_questions) * 100
        print(f"总计处理设备: 10 个")
        print(f"总计测试题数: {total_all_questions}")
        print(f"总计正确题数: {total_all_correct}")
        print(f"🎯 整体准确率: {overall_acc:.2f}%")
    print("\n🎉 All 10 devices have been processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())