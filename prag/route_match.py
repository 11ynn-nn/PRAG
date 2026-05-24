import json
import os
import time
from Cloud_Router import route_and_retrieve  # 直接导入你的云端路由核心函数

def run_router_matching():
    print("🚀 开始使用加噪向量进行云端路由匹配...\n")
    
    total_processed = 0
    
    for i in range(5,10):
        # 路径配置
        noised_file = f"noised_queries_0.1/noised_query_{i}.json"
        gt_file = f"knowledge_base_hotpot/device_{i}/questions.json"
        out_file = f"route_match_0.1/route_match_{i}.json"
        
        if not os.path.exists(noised_file):
            print(f"⚠️ 找不到加噪文件 {noised_file}，跳过该设备。")
            continue
            
        # ==========================================
        # 1. 建立 Ground Truth 映射 (提取 correct_device)
        # ==========================================
        gt_mapping = {}
        if os.path.exists(gt_file):
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                
            # 兼容带有 "test_cases" 的结构或纯列表结构
            items = gt_data.get("test_cases", []) if isinstance(gt_data, dict) else gt_data
                
            for item in items:
                q_id = str(item.get("id", item.get("question_id", "")))
                # 优先读取 device 字段，如果 JSON 里没有写死 device 字段，则默认当前文件夹对应的 device_i
                dev = item.get("device", f"device_{i}")
                gt_mapping[q_id] = dev

        # ==========================================
        # 2. 读取加噪向量并进行云端匹配
        # ==========================================
        with open(noised_file, 'r', encoding='utf-8') as f:
            noised_data = json.load(f)
            
        results = []
        print(f"[{noised_file}] 开始执行云端路由匹配，共 {len(noised_data)} 条数据...")
        
        for idx, item in enumerate(noised_data):
            q_id = str(item.get("question_id"))
            question = item.get("question")
            
            # 提取 10 个加噪向量
            noisy_vectors = []
            for j in range(1, 11):
                vec_key = f"v_query_noisy_{j}"
                if vec_key in item:
                    noisy_vectors.append(item[vec_key])
            
            if not noisy_vectors:
                print(f"  [!] ID: {q_id} 找不到加噪向量，已跳过。")
                continue
            
            # 🌟 开始云端检索计时
            start_time = time.time()
            
            # 调用 Cloud_Router 中的路由函数 (传入 10 个加噪向量)
            top_domains_with_scores, domain_summaries, routed_devices = route_and_retrieve(
                question=question,
                noisy_vectors=noisy_vectors
            )
            
            # 🌟 结束计时
            calc_time = time.time() - start_time
            
            # 提取命中的领域名列表 (Top Domains)
            hit_domains = [d[0] for d in top_domains_with_scores]
            
            # 获取正确的 device
            correct_device = gt_mapping.get(q_id, f"device_{i}")
            
            # 组装最终结果
            record = {
                "question_id": q_id,
                "question": question,
                "hit_domain": hit_domains,
                "hit_device": routed_devices,
                "calculate_time": round(calc_time, 4),
                "correct_device": correct_device
            }
            results.append(record)
            
            # 打印进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(noised_data):
                print(f"  -> 已匹配 {idx + 1}/{len(noised_data)} 条 | 最新耗时: {calc_time:.4f}s")
                
        # ==========================================
        # 3. 保存匹配结果
        # ==========================================
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 保存成功: {out_file}\n")
        total_processed += len(results)
        
    print(f"🎉 全部执行完毕！共完成 {total_processed} 条查询的云端路由模拟。")

if __name__ == "__main__":
    run_router_matching()