import os
import json
import glob
import torch
import time
from FlagEmbedding import FlagReranker

class BatchReranker:
    def __init__(self, model_path="./rerank_model"):
        print(f"正在加载 Reranker 模型: {model_path} ...")
        try:
            # 自动选择 GPU 或 CPU
            self.reranker = FlagReranker(model_path, use_fp16=torch.cuda.is_available())
            print("Reranker 加载成功！(使用 " + ("GPU FP16" if torch.cuda.is_available() else "CPU") + ")")
        except Exception as e:
            print(f"Reranker 加载失败，请检查模型路径: {e}")
            exit(1)

    def parse_retrieved_context(self, raw_text):
        """解析交替出现的 --------relations-------- 和 --------context--------"""
        relations = []
        contexts = []
        if not raw_text:
            return relations, contexts
            
        current_mode = None
        current_chunk = []
        lines = str(raw_text).split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 识别新的标识符
            if "--------relations--------" in line:
                if current_chunk and current_mode == "context":
                    contexts.append("\n".join(current_chunk))
                    current_chunk = []
                current_mode = "relations"
                continue
            elif "--------context--------" in line:
                if current_chunk and current_mode == "context":
                    contexts.append("\n".join(current_chunk))
                    current_chunk = []
                current_mode = "context"
                continue
                
            if current_mode == "relations":
                relations.append(line)
            elif current_mode == "context":
                # 处理以 【Title: 开头的文本块
                if line.startswith("【Title:"):
                    if current_chunk:
                        contexts.append("\n".join(current_chunk))
                        current_chunk = []
                current_chunk.append(line)
                
        if current_chunk and current_mode == "context":
            contexts.append("\n".join(current_chunk))
            
        return relations, contexts

    def process_and_rerank(self, query, raw_context):
        """处理单条数据，返回重排序后的结果及耗时"""
        start_total = time.time()

        # 1. 文本解析
        all_relations, all_sources = self.parse_retrieved_context(raw_context)
        
        # 2. 模型重排序 (核心计时区)
        start_rerank_core = time.time()
        
        # 精排 Sources (取 Top 3)
        top_clean_sources = []
        if all_sources:
            pairs = [[query, u] for u in all_sources]
            scores = self.reranker.compute_score(pairs)
            if isinstance(scores, float): scores = [scores]
            scored_units = sorted(zip(scores, all_sources), key=lambda x: x[0], reverse=True)
            top_clean_sources = [item[1].strip() for item in scored_units[:3]]

        # 精排 Relations (取 Top 30)
        top_clean_relations = []
        if all_relations:
            pairs = [[query, r] for r in all_relations]
            scores = self.reranker.compute_score(pairs)
            if isinstance(scores, float): scores = [scores]
            scored_rels = sorted(zip(scores, all_relations), key=lambda x: x[0], reverse=True)
            top_clean_relations = [item[1].strip() for item in scored_rels[:30]]
            
        rerank_duration = time.time() - start_rerank_core

        # 3. 组装最终结果
        relations_text = "\n".join(top_clean_relations) if top_clean_relations else "No specific relations found."
        sources_text = "\n\n".join(top_clean_sources) if top_clean_sources else "No specific text details found."
        
        reranked_context = f"--------relations--------\n{relations_text}\n\n--------context--------\n{sources_text}\n"
        
        total_duration = time.time() - start_total
        
        return {
            "reranked_context": reranked_context,
            "rerank_time": rerank_duration,
            "total_process_time": total_duration
        }

def main():
    # 基础配置：请确保脚本在 collaborative_results_0.4 文件夹的同级目录下运行
    input_base_dir = "out/out_0.4/collaborative_results"
    output_base_dir = "out/out_0.4/rerank_results"
    
    # 【核心修改】：匹配 device_0 到 device_9 下所有的 .json 文件
    # 使用递归通配符找到所有像 device_0/5a7d...json 这样的文件
    search_pattern = os.path.join(input_base_dir, "device_*", "*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"错误：在 '{input_base_dir}' 目录下未发现符合 device_*/ID.json 格式的文件！")
        return

    print(f"共检测到 {len(json_files)} 个 JSON 文件待处理...")

    processor = BatchReranker(model_path="./rerank_model")
    total_processed = 0
    
    for file_path in json_files:
        # 解析文件夹结构，确保输出时保留 device_x/文件名 的层级
        relative_path = os.path.relpath(file_path, input_base_dir)
        output_path = os.path.join(output_base_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"[{total_processed + 1}] 正在处理: {relative_path} ... ", end="", flush=True)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 执行重排序逻辑
            if isinstance(data, dict):
                query = data.get("question", "")
                raw_context = data.get("retrieved_context", "")
                
                if query and raw_context:
                    result = processor.process_and_rerank(query, raw_context)
                    
                    # 写入新字段和用时记录
                    data["reranked_context"] = result["reranked_context"]
                    data["rerank_time_seconds"] = f"{result['rerank_time']:.4f}"
                    data["rerank_total_process_time"] = f"{result['total_process_time']:.4f}"
                    
                    print(f"耗时 {data['rerank_time_seconds']}s")
                else:
                    print("跳过 (缺少字段)")
            
            # 保存至新目录
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            total_processed += 1
            
        except Exception as e:
            print(f"失败：{e}")

    print(f"\n{'='*50}")
    print(f"任务完成！总计处理 {total_processed} 个文件。")
    print(f"输出目录: {os.path.abspath(output_base_dir)}")

if __name__ == "__main__":
    main()