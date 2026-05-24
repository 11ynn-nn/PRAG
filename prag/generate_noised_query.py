import json
import os
import ollama
import asyncio
import time
from Local_Privacy_for_test import apply_noise  # 从你的原文件导入加噪函数

async def process_noised_queries(noise_level: float = 0.4):
    print(f"🌟 开始生成加噪向量，并记录单条处理耗时 (Noise Level: {noise_level})...\n")
    
    total_processed_all = 0
    
    for i in range(10):
        input_file = f"route_queries/route_query_{i}.json"
        output_file = f"noised_queries_0.4/noised_query_{i}.json"
        
        if not os.path.exists(input_file):
            continue
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data: continue
                
            noised_results = []
            print(f"[{input_file}] 开始处理 {len(data)} 条数据...")
            
            for idx, item in enumerate(data):
                q_id = item.get("question_id")
                query = item.get("question")
                if not query: continue
                
                # ==========================================
                # 🌟 开始单条 Query 计时
                # ==========================================
                query_start_time = time.time()
                
                # 1. 调用 Ollama 获取干净向量
                clean_vector = ollama.embeddings(model='nomic-embed-text', prompt=f"search_query: {query}")['embedding']
                
                # 2. 组装基础记录
                record = {
                    "question_id": q_id,
                    "question": query
                }
                
                # 3. 循环生成 10 个加噪版本
                for j in range(1, 11):
                    record[f"v_query_noisy_{j}"] = apply_noise(clean_vector, noise_level)
                
                # ==========================================
                # 🌟 结束单条 Query 计时并记录 (保留 4 位小数)
                # ==========================================
                query_time = time.time() - query_start_time
                record["calculate_time"] = round(query_time, 4)
                
                noised_results.append(record)
                    
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(noised_results, f, ensure_ascii=False, indent=4)
                
            print(f"✅ 生成完毕: {output_file}")
            total_processed_all += len(noised_results)
            
        except Exception as e:
            print(f"❌ 处理 {input_file} 时发生错误: {e}")

    print(f"\n🎉 全部完成！共生成了 {total_processed_all} 条带有耗时记录的加噪数据。")

if __name__ == "__main__":
    asyncio.run(process_noised_queries(noise_level=0.3))