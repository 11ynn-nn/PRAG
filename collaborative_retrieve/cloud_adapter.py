import numpy as np
# 导入你提供的 cloud_retrieve.py 中的类
from cloud_retrieve import VectorRouter

class BlindCloudRouter(VectorRouter):
    """
    继承自你现有的 VectorRouter，但增加了处理 '扰动向量' 的能力
    """
    def route_by_vector(self, noisy_vector, top_k=10):
        """
        [修改版] Stage 1: 基于向量直接路由，不再调用 Embedding 模型
        """
        if self.center_matrix is None: return []

        # 直接使用传入的向量
        query_vec = np.array(noisy_vector)
        
        # 矩阵运算 (相似度计算)
        scores = np.dot(self.center_matrix, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            domain = self.domain_names[idx]
            score = scores[idx]
            if score > 0.0: results.append((domain, score))
        return results

    def multicast_by_vector(self, noisy_vector, target_domains):
        """
        [修改版] Stage 2: 基于向量在 ChromaDB 中检索候选设备
        """
        if not target_domains: return []
        
        where_clause = {"domain": target_domains[0]} if len(target_domains) == 1 else {"domain": {"$in": target_domains}}
        
        # 核心修改：使用 query_embeddings 而不是 query_texts
        results = self.collection.query(
            query_embeddings=[noisy_vector], # <--- 这里的向量已经是加了扰动的
            n_results=5,  # 扩大召回，因为向量不准
            where=where_clause
        )
        
        candidates = {}
        if results['ids'] and results['ids'][0]:
            metas = results['metadatas'][0]
            dists = results['distances'][0]
            for i, meta in enumerate(metas):
                edge_id = meta['edge_id']
                # 稍微放宽阈值，因为向量有噪声 (比如从 0.45 放宽到 0.55)
                if dists[i] < 0.55: 
                    candidates[edge_id] = meta['domain']
                    
        return list(candidates.keys()) # 返回 ['edge-a2', 'edge-a1', 'edge-a3']