import os
import json
import glob
import time
from openai import OpenAI

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
    input_dir = "out/out_0.4/rerank_results"
    output_dir = "out/out_0.4/collaborative_answer"
    
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

        if query and context:
            # 推理
            import asyncio
            # 注意：此处为简化逻辑使用同步方式调用 OpenAI Client，vLLM 吞吐极高
            ans, duration = asyncio.run(engine.generate_answer(query, context))
            
            # 记录数据
            output_data = {
                "question_id": data.get("question_id", "N/A"),
                "question": query,
                "answer": ans,
                "inference_time_seconds": f"{duration:.4f}",
                "rerank_time": data.get("rerank_time_seconds", "N/A"), # 保留之前的重排用时对比
                "device": rel_path.split(os.sep)[0]
            }

            # 保存结果
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            
            print(f"[{i+1}/{len(json_files)}] 处理完成: {rel_path} | 用时: {duration:.2f}s")

    print(f"\n所有答案已保存至: {output_dir}")

if __name__ == "__main__":
    main()