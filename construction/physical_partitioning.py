import numpy as np
import pandas as pd
import os
import json
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
try:
    from config import NODE_CAPABILITIES
except ImportError:
    # 仅作为 fallback，防止单独运行此文件报错
    print("⚠️ Config not found, using empty capabilities.")
    NODE_CAPABILITIES = {}

class PhysicalPartitioner:
    def __init__(self, nodes_data=None):
        """
        初始化物理分区器
        :param nodes_data: 节点能力字典 (来自 config.py)
        """
        self.nodes_data = nodes_data if nodes_data else NODE_CAPABILITIES
        self.node_ids = sorted(list(self.nodes_data.keys())) # 确保顺序一致
        self.n = len(self.node_ids)
        self.rtt_matrix = None
        self.regions = {}

    def get_rtt_matrix(self, real_rtt_file="real_rtt_matrix.json"):
        """
        获取 RTT 矩阵：优先读取真实文件，否则基于 config 生成模拟数据。
        """
        # --- 1. 尝试读取真实 RTT 值 ---
        if os.path.exists(real_rtt_file):
            print(f"[*] 检测到真实 RTT 记录文件: {real_rtt_file}")
            try:
                with open(real_rtt_file, 'r') as f:
                    data = json.load(f)
                    # 校验文件格式是否包含所有当前节点
                    if all(nid in data for nid in self.node_ids):
                        print("✅ 成功加载真实 RTT 数据。")
                        # 构建矩阵
                        matrix = np.zeros((self.n, self.n))
                        for i, u in enumerate(self.node_ids):
                            for j, v in enumerate(self.node_ids):
                                matrix[i][j] = data[u].get(v, 9999) # 默认极大值
                        self.rtt_matrix = matrix
                        return matrix
                    else:
                        print("⚠️ 真实 RTT 数据节点不匹配，回退到模拟模式。")
            except Exception as e:
                print(f"❌ 读取真实 RTT 文件失败: {e}，回退到模拟模式。")
        
        # --- 2. 回退到模拟模式 (基于 config 中的 'region' 标签) ---
        print("[*] 未找到真实 RTT 或读取失败，生成模拟 RTT 矩阵...")
        matrix = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                u = self.node_ids[i]
                v = self.node_ids[j]
                
                region_u = self.nodes_data[u].get('region', 'UNKNOWN')
                region_v = self.nodes_data[v].get('region', 'UNKNOWN')
                
                if region_u == region_v:
                    # 同区域：低延迟 (5ms - 20ms)
                    rtt = np.random.normal(12, 3) 
                else:
                    # 跨区域：高延迟 (100ms - 200ms)
                    rtt = np.random.normal(150, 20)
                
                # 保持对称性
                matrix[i][j] = matrix[j][i] = max(1.0, rtt) # 至少1ms
                
        self.rtt_matrix = matrix
        return matrix

    def run_partitioning(self, tau=50):
        """
        执行 GSAP 物理分区
        :param tau: 延迟阈值 (ms)，低于此值的节点将被聚为同一个 Region
        """
        if self.rtt_matrix is None:
            self.get_rtt_matrix()

        print(f"[*] 执行凝聚型层次聚类 (Threshold tau={tau}ms)...")
        
        # 1. 压缩距离矩阵 (scipy linkage 要求)
        condensed_dist = squareform(self.rtt_matrix)
        
        # 2. 聚类 (使用平均连接法 UPGMA，适合物理距离)
        Z = linkage(condensed_dist, method='average')
        
        # 3. 截断聚类树得到 Cluster ID (1, 2, 3...)
        cluster_labels = fcluster(Z, t=tau, criterion='distance')
        
        # 4. 映射为 Region 名称 (region-a, region-b...)
        # 按照 Cluster 中包含的第一个节点的字母序来决定 region-a 还是 b，保证确定性
        unique_labels = sorted(list(set(cluster_labels)))
        
        # 建立映射表: Cluster ID -> Region Name
        cluster_to_name = {}
        # 为了让结果好看（比如 region A 的节点真的叫 region-a），我们尝试匹配
        # 这里简单起见，直接按聚类结果出现的顺序命名 a, b, c...
        # 也可以做一个高级逻辑：统计簇内节点 'region' 标签的众数
        
        ascii_offset = 0
        for lab in unique_labels:
            # 找到该簇的所有节点
            indices = np.where(cluster_labels == lab)[0]
            member_ids = [self.node_ids[i] for i in indices]
            
            # 尝试推断该 Region 的“真名” (上帝视角辅助)
            # 统计成员中 config['region'] 出现最多的 (e.g., 'A')
            tags = [self.nodes_data[mid].get('region', 'X') for mid in member_ids]
            most_common_tag = max(set(tags), key=tags.count)
            
            if most_common_tag.isalpha() and len(most_common_tag) == 1:
                final_name = f"region-{most_common_tag.lower()}"
            else:
                final_name = f"region-{chr(97 + ascii_offset)}"
                ascii_offset += 1
            
            cluster_to_name[lab] = final_name

        # 5. 生成最终结果
        self.regions = {}
        print("\n=== GSAP 物理分区结果 ===")
        print(f"{'Node ID':<20} | {'Sim/Real RTT Avg':<15} | {'Assigned Region'}")
        print("-" * 60)
        
        for i, node_id in enumerate(self.node_ids):
            lab = cluster_labels[i]
            region_name = cluster_to_name[lab]
            self.regions[node_id] = region_name
            
            # 计算该节点与同 Region 其他节点的平均 RTT (用于展示)
            indices = np.where(cluster_labels == lab)[0]
            if len(indices) > 1:
                avg_rtt = np.mean([self.rtt_matrix[i][x] for x in indices if x != i])
                rtt_str = f"{avg_rtt:.1f} ms"
            else:
                rtt_str = "N/A (Single)"
                
            print(f"{node_id:<20} | {rtt_str:<15} | {region_name}")

        return self.regions

# --- 供外部调用 ---
def perform_physical_partitioning():
    partitioner = PhysicalPartitioner(NODE_CAPABILITIES)
    # 可以在这里指定真实文件路径，如果不存在则自动模拟
    return partitioner.run_partitioning(tau=50)

if __name__ == "__main__":
    perform_physical_partitioning()