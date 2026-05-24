import os
import json
import glob
import time
from openai import OpenAI

NOISE_LEVEL=0.8

class VLLMInferenceWrapper:
    def __init__(self, model_name="qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1"):
        self.model = model_name
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    async def generate_answer(self, query, reranked_context):
        sys_prompt = """---Role---
You are an expert AI assistant. Your task is to answer user queries accurately by ONLY using the provided **Context**.

---Knowledge Structure---
The Context consists of two parts:
1. --------relations--------: Structural triples showing entity connections.
2. --------context--------: Detailed factual paragraphs with titles.

---Instructions---
1. **NOISE FILTERING**: IGNORE all text that does not explicitly match the entities in the User Query.
2. **Step-by-Step Reasoning**: Analyze Query Intent -> Extract Facts from both Relations and Paragraphs -> Synthesize.
3. **GROUNDING**: Answer ONLY based on the provided Data. If the information is not present, state that you cannot answer.

---Response Rules---
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Document Chunks (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base.
"""

        user_prompt = f"""---Context---
{reranked_context}

---User Query---
{query}

---Instruction---
Based ONLY on the Context Data above (both relations and text), answer the query. 
---Answer---
"""
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )
            answer = response.choices[0].message.content.strip()
            inference_time = time.time() - start_time
            return answer, inference_time
        except Exception as e:
            print(f"\n[Error] vLLM 调用失败: {e}")
            return "Error during generation.", 0.0

def main():
    input_dir = f"ppec/noise_{NOISE_LEVEL}/rerank_results_1"
    output_dir = f"ppec/noise_{NOISE_LEVEL}/collaborative_answer_1"
    
    # 🌟 修改点 1: 加载并索引
    results_path = f"ppec/noise_{NOISE_LEVEL}/results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        # 将 id 转为字符串以防匹配失败
        results_map = {str(item.get("id")): item for item in results_data}
    else:
        print(f"警告：找不到 {results_path}，将跳过同步更新。")
        results_map = {}

    # 获取所有重排后的 JSON 文件
    search_pattern = os.path.join(input_dir, "device_*", "*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"未在 {input_dir} 找到文件。")
        return

    # 初始化推理引擎
    engine = VLLMInferenceWrapper()
    print(f"开始批量推理，共 {len(json_files)} 个文件...")

    for i, file_path in enumerate(json_files):
        # 路径处理
        rel_path = os.path.relpath(file_path, input_dir)
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        query = data.get("question", "")
        context = data.get("reranked_context", "")
        q_id = str(data.get("id", "N/A"))

        if query and context:
            # 推理
            import asyncio
            # 注意：此处为简化逻辑使用同步方式调用 OpenAI Client，vLLM 吞吐极高
            ans, duration = asyncio.run(engine.generate_answer(query, context))
            
            # 🌟 修改点 2: 同步更新 results_map 中的 collaborative_inference_time_1 和 collaborative_answer_1
            if q_id in results_map:
                results_map[q_id]["collaborative_inference_time_1"] = round(duration, 4)
                results_map[q_id]["collaborative_answer_1"] = ans

            # 记录数据
            output_data = {
                "id": q_id,
                "question": query,
                "answer": ans,
                "inference_time_seconds": f"{duration:.4f}",
                "rerank_time": data.get("rerank_time_seconds", "N/A"), # 保留之前的重排用时对比
                "device": rel_path.split(os.sep)[0]
            }

            # 保存结果
            with open(save_path, 'w+', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            print(f"[{i+1}/{len(json_files)}] 处理完成: {rel_path} | 用时: {duration:.2f}s")

    # 🌟 修改点 3: 处理完毕后，将更新后的数据写回 ppec/results.json
    if results_map:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)
        print(f"\n💾 已同步更新 {results_path} 中的 collaborative_inference_time_1 和 collaborative_answer_1 字段。")

    print(f"\n所有答案已保存至: {output_dir}")

if __name__ == "__main__":
    main()