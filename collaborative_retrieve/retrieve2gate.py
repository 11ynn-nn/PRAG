import json
import os
import asyncio
import numpy as np
import time
import ollama
import traceback

# === 导入 LightRAG ===
from lightrag import LightRAG, QueryParam
# 兼容不同版本的导入路径
from lightrag.utils import EmbeddingFunc


# ================= 配置部分 =================
DATASET_PATH = "test_qa_dataset.json"      # 你的数据集
WORKING_DIR = "servers_data/edge-a1"    # 你的知识库路径
OUTPUT_FILE = "gate_test.json"  # 结果保存路径
NODE_ID = "edge-a1"

LLM_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# ================= Ollama 接口封装 =================
async def ollama_model_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 获取 kwargs 中的 temperature，如果没有则默认 0.1
    # 门控判断时通常需要更低的 temperature (0.0) 以保证稳定性
    temp = kwargs.get("temperature", 0.1)

    try:
        response = ollama.chat(
            model=LLM_MODEL, 
            messages=messages, 
            options={"temperature": temp, "num_ctx": 32000}
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"LLM Error: {e}")
        return ""

async def ollama_embedding_func(texts: list[str]) -> np.ndarray:
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        embeddings = response["embeddings"]
        return np.array(embeddings)
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros((len(texts), EMBEDDING_DIM))

# ================= 核心类: EdgeNode_Retrieve (含门控) =================
class EdgeNode_Retrieve:
    def __init__(self, node_id, working_dir):
        self.node_id = node_id
        self.working_dir = working_dir
        self.rag = None
        self.is_ready = False

    async def initialize(self):
        """初始化 LightRAG 引擎"""
        print(f"[{self.node_id}] Initializing LightRAG engine...")
        if not os.path.exists(self.working_dir):
            raise FileNotFoundError(f"Directory {self.working_dir} not found.")

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                func=ollama_embedding_func
            )
        )
        
        # 显式初始化存储 (防止 __aenter__ 错误)
        print(f"[{self.node_id}] Mounting storages (Async)...")
        if hasattr(self.rag, 'initialize_storages'):
            await self.rag.initialize_storages()
        
        self.is_ready = True
        print(f"[{self.node_id}] System Ready.")

    async def _evaluate_confidence(self, question: str, response: str) -> bool:
        """
        [优化版门控] 置信度评估器
        目标：准确拦截“不知道”的情况，但放行“基于文档回答”的情况。
        """
        # === 1. 关键词快筛 (精简版) ===
        # 只保留明显的“拒答”或“偷跑”信号
        # 注意：不再包含 "provided context" 这种中性词
        negative_signals = [
            "general knowledge", "common knowledge", # 偷跑信号
            "not mentioned", "no mention",           # 拒答信号
            "no information", "insufficient information",
            "couldn't find", "cannot find",
            "does not provide", "doesn't provide",
            "outside of the context", "training data"
        ]
        
        response_lower = response.lower()
        for kw in negative_signals:
            if kw in response_lower:
                # 为了防止误杀（比如 "It is not mentioned in X, but Y says..."），
                # 我们可以让快筛稍微宽松一点，或者只作为强烈暗示。
                # 但为了效率，这里我们还是直接拦截，但在 Prompt 里做最终把关可能更稳。
                # 现在的列表已经去掉了中性词，误杀率会大大降低。
                # print(f"  [Gate Fast-Fail] Detected negative signal: '{kw}'")
                return False

        # === 2. LLM 深度裁判 (Deep Judge) ===
        # 允许合理的推断，但严禁瞎编
        judge_prompt = (
            f"Task: Check if the AI response successfully answers the query using the provided context.\n\n"
            f"Query: \"{question}\"\n"
            f"Response: \"{response}\"\n\n"
            f"Evaluation Rules:\n"
            f"1. If the response says it CANNOT answer or the info is missing -> Output 'UNCERTAIN'.\n"
            f"2. If the response admits using 'general knowledge' or 'common knowledge' instead of the context -> Output 'UNCERTAIN'.\n"
            f"3. If the response provides an answer (even if it infers it from related facts in the context) -> Output 'CONFIDENT'.\n"
            f"4. Do NOT punish the model for saying 'Based on the provided context'. That is good.\n\n"
            f"Verdict (CONFIDENT / UNCERTAIN):"
        )

        try:
            # 温度设为 0 以保持冷静
            judgment = await ollama_model_complete(judge_prompt, temperature=0.0)
            judgment = judgment.strip().upper()
            
            if "UNCERTAIN" in judgment:
                return False
            return True
            
        except Exception as e:
            print(f"Judge Error: {e}")
            return True

    async def infer_and_judge(self, question: str):
        """
        执行推理 + 评估
        """
        if not self.is_ready:
            return {"error": "Node not initialized"}

        start_total = time.time()
        
        # --- Step 1: LightRAG 生成 (RAG Inference) ---
        start_rag = time.time()
        try:
            # 使用 hybrid 模式获取最佳回答
            prediction = await self.rag.aquery(question, param=QueryParam(mode="hybrid"))
        except Exception as e:
            prediction = f"Error: {str(e)}"
        rag_duration = time.time() - start_rag
        
        # --- Step 2: 置信度评估 (Confidence Judge) ---
        start_judge = time.time()
        is_confident = await self._evaluate_confidence(question, prediction)
        judge_duration = time.time() - start_judge
        
        # --- Step 3: 决策 (Decision) ---
        decision = "LOCAL" if is_confident else "COLLABORATE"
        
        total_duration = time.time() - start_total
        
        if decision == "LOCAL":
            return {
                "answer": prediction,
                "decision": decision,
                "latency_metrics": {
                    "rag": round(rag_duration, 4),
                    "judge": round(judge_duration, 4),
                    "total": round(total_duration, 4)
            }
        }
        else:
            return {
                "initial_answer": prediction,
                "decision": decision,
                "latency_metrics": {
                    "rag": round(rag_duration, 4),
                    "judge": round(judge_duration, 4),
                    "total": round(total_duration, 4)
            }
        }

# ================= 批量测试逻辑 =================
async def run_batch_test():
    # 1. 加载数据集
    if not os.path.exists(DATASET_PATH):
        print(f"Error: 找不到数据集文件 {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 2. 初始化边缘节点服务
    edge_node = EdgeNode_Retrieve(NODE_ID, WORKING_DIR)
    await edge_node.initialize()

    results = []
    print(f"\n🚀 开始批量测试 (含门控机制): 共 {len(dataset)} 个问题")
    print("-" * 60)

    for i, item in enumerate(dataset):
        qid = item['id']
        question = item['question']
        ground_truth = item['ground_truth']
        source_node = item.get('source_node', 'unknown')
        
        print(f"[{i+1}/{len(dataset)}] QID:{qid} (Src:{source_node}) Processing...")
        # print(f"  Q: {question[:60]}...")
        
        # === 调用核心接口 ===
        output = await edge_node.infer_and_judge(question)
        
        # 提取结果
        prediction = output['prediction']
        decision = output['decision']
        latency = output['latency_metrics']
        
        # 打印简报
        print(f"  ✅ Decision: {decision}")
        print(f"  ⏱️  Latency: {latency['total']}s (RAG: {latency['rag']}s + Judge: {latency['judge']}s)")
        if decision == "LOCAL":
             print(f"  💡 Answer: {prediction[:80].replace(chr(10), ' ')}...")
        else:
             print(f"  ⚠️  Reason: Low Confidence (Refused to answer)")
        print("-" * 60)
        
        # 构建结果记录
        result_item = item.copy()
        result_item.update({
            "prediction": prediction,
            "decision": decision,
            "latency_rag": latency['rag'],
            "latency_judge": latency['judge'],
            "latency_total": latency['total']
        })
        results.append(result_item)

    # 3. 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 测试完成！结果已保存至: {OUTPUT_FILE}")
    
    # 简单统计
    local_count = sum(1 for r in results if r['decision'] == "LOCAL")
    cloud_count = sum(1 for r in results if r['decision'] == "COLLABORATE")
    print(f"📊 统计: 本地解决: {local_count}, 转发云端: {cloud_count}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_batch_test())