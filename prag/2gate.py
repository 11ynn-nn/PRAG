import os
import json
import asyncio
import time
from pathlib import Path
from openai import OpenAI  # <--- 新增的导入

# === 你提供的 vLLM 包装器 ===
class VLLMInferenceWrapper:
    def __init__(self, model_name="qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1"):
        self.model = model_name
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

# === 评估器主类 ===
class AnswerEvaluator:
    def __init__(self, client, model="gpt-3.5-turbo"):
        self.client = client
        self.model = model

    async def _evaluate_confidence(self, question: str, response: str) -> bool:
        """
        置信度评估器
        返回: True (自信/包含答案), False (不确定/拒答)
        """
        # === 1. 关键词快筛 (精简版) ===
        negative_signals = [
            "general knowledge", "common knowledge", # 偷跑信号
            "not mentioned", "no mention",           # 拒答信号
            "no information", "insufficient information",
            "couldn't find", "cannot find",
            "does not provide", "doesn't provide",
            "outside of the context", "training data",
            "i cannot answer"
        ]
        
        response_lower = str(response).lower()
        for kw in negative_signals:
            if kw in response_lower:
                print(f"   [Gate Fast-Fail] Detected negative signal: '{kw}'")
                return False

        # === 2. LLM 深度裁判 (Deep Judge) ===
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
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0, # 温度设为 0 以保持冷静
                max_tokens=10
            )
            judgment = completion.choices[0].message.content.strip().upper()
            
            print(f"   [Gate Judge] Verdict: {judgment}")
            
            if "UNCERTAIN" in judgment:
                return False
            return True
            
        except Exception as e:
            print(f"   [Gate Error] {e}")
            return True # 默认放行，避免死循环

    async def process_directory(self, input_dir: str, output_dir: str):
        """
        遍历目录，读取 JSON，评估 answer，并输出到新目录，同时记录用时
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        json_files = list(input_path.rglob('*.json'))
        
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return

        print(f"Found {len(json_files)} JSON files. Starting evaluation...\n")
        
        total_start_time = time.time()
        total_items_processed = 0

        for file_path in json_files:
            print(f"Processing: {file_path}")
            file_start_time = time.time() 
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"  [Error] Failed to parse {file_path}. Skipping.")
                    continue

            is_list = isinstance(data, list)
            items = data if is_list else [data]

            for idx, item in enumerate(items):
                question = item.get("question", "")
                answer = item.get("answer", "")

                if not answer:
                    print(f"  [Warning] Item {idx} missing 'answer' field.")
                    continue

                # 调用评估逻辑
                is_confident = await self._evaluate_confidence(question, answer)
                item["is_confident"] = is_confident
                total_items_processed += 1

            rel_path = file_path.relative_to(input_path)
            save_path = output_path / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            final_data = items if is_list else items[0]
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
                
            file_end_time = time.time()
            print(f"Saved to: {save_path} (File took: {file_end_time - file_start_time:.2f}s)\n")

        total_end_time = time.time()
        elapsed_time = total_end_time - total_start_time
        
        print("-" * 40)
        print("🎉 Evaluation Complete!")
        print(f"Total files processed: {len(json_files)}")
        print(f"Total items evaluated: {total_items_processed}")
        print(f"Total time elapsed: {elapsed_time:.2f} seconds")
        if total_items_processed > 0:
            print(f"Average time per item: {elapsed_time / total_items_processed:.2f} seconds")
        print("-" * 40)

# === 启动脚本 ===
async def main():
    # 1. 实例化你的 vLLM 包装器
    vllm = VLLMInferenceWrapper(
        model_name="qwen/Qwen2.5-7B-Instruct", 
        base_url="http://localhost:8000/v1"
    )
    
    # 2. 将包装器中的 client 和 model 传给评估器
    evaluator = AnswerEvaluator(client=vllm.client, model=vllm.model)
    
    # 修改这里：更新为正确的输入路径和对应的输出路径
    input_directory = "local_reanswer/noise_0.4/"
    output_directory = "local_reanswer_gate/noise_0.4/"
    
    await evaluator.process_directory(input_directory, output_directory)

if __name__ == "__main__":
    asyncio.run(main())