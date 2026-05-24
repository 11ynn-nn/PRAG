# le_rag_core.py
import numpy as np
from config import *

class LERAGProtocol:
    """
    LE-RAG 协议核心算法实现
    对应论文中的 Observation, Valuation, Arbitration, Transition 阶段
    """

    @staticmethod
    def calculate_icm(caps, topology_centrality=0.5):
        """
        计算内部能力指标 (ICM)
        Formula: ICM(v) = w_io*IO + w_acc*ACC + w_tc*TC - (w_m*M + w_j*J)
        
        参数映射:
        - IO  <- caps['net'] (网络带宽)
        - ACC <- 0.7*caps['gpu'] + 0.3*caps['cpu'] (向量计算主要靠GPU)
        - TC  <- topology_centrality (外部传入)
        - M   <- 1.0 - caps['mem'] (内存净空 = 1 - 已用)
        - J   <- 模拟值 (基于 net 性能的反向波动)
        """
        io_score = caps['net']
        acc_score = 0.7 * caps['gpu'] + 0.3 * caps['cpu']
        tc_score = topology_centrality
        
        # 资源净空 (数值越大越危险，所以公式里是减去它，这里 M 代表"已用/饱和度")
        # 论文原文: "规避资源饱和"，通常指 Load。这里假设 mem 是利用率。
        # 修正: 论文公式是 - (w_m * M)，若 M 代表 Headroom(剩余)，则应是 +。
        # 假设: 论文中的 M 代表 "饱和度/负载" (Metric of Saturation)。
        m_saturation = caps['mem'] 
        
        # 抖动 J: 假设网络越差，抖动越大
        j_jitter = (1.0 - caps['net']) * 0.2 

        icm_val = (W_IO * io_score + 
                   W_ACC * acc_score + 
                   W_TC * tc_score - 
                   (W_M * m_saturation + W_J * j_jitter))
        return icm_val

    @staticmethod
    def calculate_epm_pairwise(node_u_caps, node_v_caps, history_trust=1.0):
        """
        计算节点 u 对节点 v 的单向偏好 phi(u, v)
        Formula: phi = a_d*D^-1 + a_q*Q + a_st*ST + a_tr*TR
        """
        # 1. 距离 D^-1: 假设同一区域距离为 0.1(分高)，跨区域为 0.8(分低)
        # 这里简化为基于网络性能的虚拟距离
        dist_inv = (node_u_caps['net'] + node_v_caps['net']) / 2.0
        
        # 2. 链路质量 Q: 取两端网络短板
        link_quality = min(node_u_caps['net'], node_v_caps['net'])
        
        # 3. 稳定性 ST: 模拟为 CPU 稳定度 (1 - cpu_load)
        stability = 1.0 - node_v_caps['cpu']
        
        # 4. 信任度 TR: 历史交互记录
        trust = history_trust

        phi = (ALPHA_D * dist_inv + 
               ALPHA_Q * link_quality + 
               ALPHA_ST * stability + 
               ALPHA_TR * trust)
        return phi

    @staticmethod
    def calculate_final_score(icm_score, epm_score):
        """
        Score(v) = ICM(v) + beta * EPM(v)
        """
        return icm_score + BETA * epm_score

    @staticmethod
    def check_handover_condition(score_best, score_current, real_index_size_mb):
        """
        懒惰切换判决 (Lazy Election)
        Condition: Score(Best) - Score(Curr) > lambda * C_mig(|I_global|)
        
        参数:
        - real_index_size_mb: 必须传入通过 file_ops 计算出的真实大小
        """
        if real_index_size_mb is None or real_index_size_mb < 0:
            real_index_size_mb = 0.0
            
        # C_mig 成本函数: 
        # 假设带宽限制下，每 100MB 传输会造成 0.05 的效用损失 (可根据实际网络测试调整)
        # 例如: 500MB -> Cost = 0.25
        cost_mig = (real_index_size_mb / 100.0) * 0.05
        
        threshold = LAMBDA * cost_mig
        diff = score_best - score_current
        
        is_switch = diff > threshold
        return is_switch, diff, threshold