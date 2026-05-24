import os
import glob
import json
import requests

# ==========================================
# 1. 全局配置与状态区
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b" 
BASE_DIR = "servers_data" # 你的数据根目录

# 预设的“种子领域”，大模型会优先使用这些词，避免无中生有
established_domains = {
    "Technology", "Health", "Finance", "Music", "Film", "Literature",
    "Sports", "Art", "Society", "Game", "Law"
}

# 强制映射字典 (别名 -> 标准大类)
# 你可以在这里随意添加你不想看到的细分词汇
DOMAIN_ALIASES = {
    "Mental Health": "Health",
    "Health Sciences": "Health",
    "Medicine": "Health",
    "Healthcare": "Health",
    "Medical": "Health",
    "Information Technology": "Technology",
    "It": "Computer Science",
    "Software": "Computer Science",
    "Gaming": "Game",
    "Games": "Game",
    "Business": "Finance",
    "Economics": "Finance",
    "Movies": "Film"
}

# ==========================================
# 2. LLM 调用与分类逻辑
# ==========================================
def call_ollama(prompt):
    payload = {
        "model": MODEL_NAME, 
        "prompt": prompt, 
        "stream": False,
        "options": {"temperature": 0.0} # 温度设为0，保证分类的确定性和死板
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload)
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"调用 Ollama 出错: {e}")
        return "General"

def classify_subgraph_dynamic(summary_text):
    global established_domains
    
    # 将当前的领域集合转成字符串供大模型参考
    domains_str = ", ".join(established_domains)
    
    prompt = (
        f"You are a strict knowledge classifier. Read the following summary of a knowledge graph fragment:\n"
        f"\"{summary_text}\"\n\n"
        f"Currently established BROAD domain categories are: [{domains_str}].\n\n"
        f"Task: Classify this text into 1 to 3 broad domain categories.\n"
        f"CRITICAL RULES:\n"
        f"1. You MUST prioritize using the established categories if they are a reasonable fit.\n"
        f"2. Only invent a NEW category if the text completely falls outside the existing ones. If you create a new one, make it a BROAD, single-word or two-word noun.\n"
        f"Output ONLY a comma-separated list of category names. Do not add any other words or punctuation."
    )
    
    response_text = call_ollama(prompt)
    
    # 1. 拆分并初步清洗
    raw_categories = response_text.split(',')
    cleaned_domains = set()
    
    for cat in raw_categories:
        # 去除标点、两端空格并统一格式为 Title Case
        clean_cat = cat.replace(".", "").strip().title()
        if not clean_cat:
            continue
            
        # 2. 经过硬编码的映射字典过滤 (拦截同义词)
        # 如果这个词在别名表里，就替换成标准词；否则保持原样
        standard_cat = DOMAIN_ALIASES.get(clean_cat, clean_cat)
        cleaned_domains.add(standard_cat)
    
    # 3. 动态扩充我们的全局领域池 (以便下一个循环使用)
    established_domains.update(cleaned_domains)
    
    # 容错保底
    if not cleaned_domains:
        return ["General"]
        
    return list(cleaned_domains)

# ==========================================
# 3. 云端聚合核心管线
# ==========================================
def aggregate_and_classify_all_edges(base_dir=BASE_DIR):
    print("=== Step 1: 正在从所有边缘节点拉取并聚合数据 ===")
    global_summary_pool = []
    
    # 匹配 10 个边缘设备的 json 文件 (如 servers_data/edge-a1/edge-a1_summary.json)
    search_pattern = os.path.join(base_dir, "edge-*", "*_summary.json")
    summary_files = glob.glob(search_pattern)
    
    if not summary_files:
        print(f"未能在 {base_dir} 目录下找到边缘节点的 JSON 文件，请检查路径。")
        return
        
    for file_path in summary_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                edge_id = data.get("edge_device", os.path.basename(os.path.dirname(file_path)))
                subgraphs = data.get("subgraphs", [])
                
                for sg in subgraphs:
                    if "summary" in sg and sg["summary"]:
                        global_summary_pool.append({
                            "source_edge": edge_id,
                            "subgraph_id": sg.get("subgraph_id", "unknown_id"),
                            "type": sg.get("type", "unknown_type"),
                            "node_count": sg.get("node_count", 0),
                            "summary": sg["summary"]
                        })
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")

    total_summaries = len(global_summary_pool)
    print(f"物理聚合完成！共加载 {len(summary_files)} 个设备，提取 {total_summaries} 个子图摘要。")
    print("\n=== Step 2: 正在云端进行全局多维度语义划分 (应用动态防扩散策略) ===")

    cloud_semantic_communities = {}
    
    for i, item in enumerate(global_summary_pool, 1):
        summary_text = item["summary"]
        source_edge = item["source_edge"]
        subgraph_id = item["subgraph_id"]
        
        # 调用智能分类
        domains = classify_subgraph_dynamic(summary_text)
        print(f"[{i:03d}/{total_summaries}] [{source_edge}] {subgraph_id} -> {', '.join(domains)}")
        
        # 将子图挂载到所有被命中的领域下 (实现 Overlapping Communities)
        for domain in domains:
            if domain not in cloud_semantic_communities:
                cloud_semantic_communities[domain] = []
                
            cloud_semantic_communities[domain].append({
                "source_edge": source_edge,
                "subgraph_id": subgraph_id,
                "node_count": item["node_count"],
                "summary": summary_text
            })

    # 4. 结果落盘
    output_path = os.path.join(base_dir, "cloud_semantic_communities.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            final_output = {
                "system_status": "aggregated",
                "total_edges_processed": len(summary_files),
                "total_subgraphs_processed": total_summaries,
                "total_semantic_domains_generated": len(cloud_semantic_communities),
                "final_domain_list": list(cloud_semantic_communities.keys()),
                "domains": cloud_semantic_communities
            }
            json.dump(final_output, f, indent=4, ensure_ascii=False)
            
        print(f"\n聚合与分类全部完成！")
        print(f"最终收敛至 {len(cloud_semantic_communities)} 个宏观领域。")
        print(f"结果已保存至: {output_path}")
        
    except Exception as e:
        print(f"保存云端聚合结果时出错: {e}")

if __name__ == "__main__":
    aggregate_and_classify_all_edges()