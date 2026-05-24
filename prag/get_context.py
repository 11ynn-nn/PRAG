import os
import json
import asyncio
import time
# 导入你的异步检索核心函数
from CR import run_architecture

async def process_batch(batch_idx):
    noised_file = f"noised_queries_search_1/noised_query_search_0.1/noised_query_{batch_idx}.json"
    route_file = f"out/out_0.1/route_match/route_match_{batch_idx}.json"
    output_dir = f"out/out_0.1/collaborative_results/device_{batch_idx}"

    if not os.path.exists(noised_file) or not os.path.exists(route_file):
        print(f"⚠️ [跳过] 找不到批次 {batch_idx} 的输入文件。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载路由匹配数据
    with open(route_file, 'r', encoding='utf-8') as f:
        route_data = json.load(f)
    
    # 兼容路由数据的 ID 获取 (增加 question_id 检查)
    route_dict = {}
    if isinstance(route_data, list):
        for item in route_data:
            r_id = item.get("question_id") or item.get("id") or item.get("_id")
            if r_id: route_dict[str(r_id)] = item.get("hit_device", [])
    elif isinstance(route_data, dict):
        for k, v in route_data.items():
            r_id = v.get("question_id") or v.get("id") or v.get("_id") or k
            route_dict[str(r_id)] = v.get("hit_device", [])

    # 2. 加载加噪查询数据
    with open(noised_file, 'r', encoding='utf-8') as f:
        noised_data = json.load(f)
    
    # 兼容加噪数据的列表/字典结构
    noised_list = []
    if isinstance(noised_data, list):
        noised_list = noised_data
    elif isinstance(noised_data, dict):
        for k, v in noised_data.items():
            if isinstance(v, dict):
                # 如果字典里没有明确的 id 字段，把最外层的 key 当作 id
                if not any(key in v for key in ["question_id", "id", "_id"]):
                    v["question_id"] = k
                noised_list.append(v)

    # 3. 遍历执行
    processed_count = 0
    for query_item in noised_list:
        # ==========================================
        # 核心修复点：精确提取 question_id
        # ==========================================
        q_id = query_item.get("question_id") or query_item.get("id") or query_item.get("_id")
        
        if not q_id:
            print(f"  [!] 警告: 发现无法解析 ID 的数据条目，已跳过。")
            continue
            
        q_id = str(q_id) # 统一转为字符串处理

        # 提取 1-5 的加噪向量
        noisy_vectors = []
        for j in range(1, 6):
            vec_key = f"v_query_noisy_{j}"
            if vec_key in query_item:
                noisy_vectors.append(query_item[vec_key])
        
        if len(noisy_vectors) != 5:
            print(f"  [!] Query ID {q_id} 没有完整的 5 个加噪向量，跳过。")
            continue

        hit_devices = route_dict.get(q_id, [])

        start_time = time.perf_counter()

        # 调用检索核心函数
        retrieved_context = await run_architecture(
            target_devices=hit_devices,
            noisy_vectors=noisy_vectors,
            sim_threshold=0.75
        )

        # ==========================================
        # 新增：结束计时并计算耗时 (单位：秒)
        # ==========================================
        end_time = time.perf_counter()
        retrieve_time = end_time - start_time

        # 构建并保存结果
        result_content = {
            "question_id": q_id,
            "search_device": hit_devices,
            "retrieved_context": retrieved_context,
            "Retrieve_time": retrieve_time  # <--- 新增：将耗时保存到 JSON
        }

        output_filepath = os.path.join(output_dir, f"{q_id}.json")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(result_content, f, ensure_ascii=False, indent=4)
        
        processed_count += 1

    print(f"✅ 批次 {batch_idx} 处理完成，成功生成了 {processed_count} 个结果文件。")


async def main():
    for i in range(10):
        print(f"\n🚀 开始处理批次: {i}")
        await process_batch(i)
    print("\n🎉 所有检索任务执行完毕！")

if __name__ == "__main__":
    asyncio.run(main())