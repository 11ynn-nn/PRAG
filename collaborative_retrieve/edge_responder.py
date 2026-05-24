import json
import os
import numpy as np
import asyncio
import math
from collaborative_retrieve.privacy_utils import PrivacyUtils

# === 导入 LightRAG ===
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
import ollama

# 配置
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

async def ollama_embedding_func(texts: list[str]) -> np.ndarray:
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        embeddings = response["embeddings"]
        return np.array(embeddings)
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros((len(texts), EMBEDDING_DIM))

async def dummy_llm(prompt, **kwargs): 
    return ""

class EdgeResponderService:
    def __init__(self, node_id, working_dir):
        self.node_id = node_id
        self.working_dir = working_dir
        self.rag = None
        self.is_ready = False

    async def initialize(self):
        print(f"[{self.node_id}] Initializing LightRAG Engine...")
        if not os.path.exists(self.working_dir):
            print(f"Path {self.working_dir} not found.")
            return

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=dummy_llm, 
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBEDDING_DIM,
                max_token_size=8192,
                func=ollama_embedding_func
            )
        )
        
        try:
            if hasattr(self.rag, 'initialize_storages'):
                await self.rag.initialize_storages()
        except Exception as e:
            print(f"[{self.node_id}] Storage init warning: {e}")
            
        self.is_ready = True
        print(f"[{self.node_id}] Ready.")

    async def handle_blind_query(self, noisy_vector, pub_key_pem):
        try:
            # 1. 格式转换
            query_vec = np.array(noisy_vector).flatten()
            if query_vec.shape[0] != EMBEDDING_DIM:
                if query_vec.shape[0] > EMBEDDING_DIM:
                    query_vec = query_vec[:EMBEDDING_DIM]
                else:
                    query_vec = np.pad(query_vec, (0, EMBEDDING_DIM - query_vec.shape[0]))
            
            query_vec_list = query_vec.tolist()

            # 2. 直接调用底层 _client 查询
            if hasattr(self.rag.chunks_vdb, "_client"):
                # 同步调用
                results = self.rag.chunks_vdb._client.query(query_vec_list, top_k=40)
            else:
                print(f"Warning: _client not found, trying standard query.")
                results = await self.rag.chunks_vdb.query(query_vec_list, top_k=40)
            
            scored_packets = []
            
            # [调试] 打印第一个结果的 __metrics__ 结构，确认它是 float 还是 dict
            for res in results:
                content = res.get('content') or res.get('text')
                
                # KV 回查
                if not content and 'id' in res:
                    try:
                        doc = await self.rag.chunks_kv_storage.get_by_id(res['id'])
                        if doc: content = doc.get('content')
                    except: pass

                if content:
                    clean_content = content.replace('\n', ' ').strip()
                    
                    # [核心修复] 从 __metrics__ 提取分数
                    raw_score = 0.0
                    metrics = res.get('__metrics__')
                    
                    if metrics is not None:
                        # 情况A: metrics 直接是数值 (距离)
                        if isinstance(metrics, (int, float)):
                            # 假设是 Cosine Distance (0~2)，我们需要 Similarity (越大越好)
                            # Similarity = 1 - Distance
                            raw_score = 1.0 - float(metrics)
                        # 情况B: metrics 是字典
                        elif isinstance(metrics, dict):
                            if 'cosine_similarity' in metrics:
                                raw_score = float(metrics['cosine_similarity'])
                            elif 'score' in metrics:
                                raw_score = float(metrics['score'])
                            elif 'distance' in metrics:
                                raw_score = 1.0 - float(metrics['distance'])
                            else:
                                # 盲猜第一个值
                                try:
                                    raw_score = 1.0 - float(list(metrics.values())[0])
                                except: pass
                    else:
                        # 尝试旧字段
                        raw_score = res.get('score', res.get('distance', 0.0))

                    # 清洗 nan
                    try:
                        score = float(raw_score)
                        if math.isnan(score) or math.isinf(score):
                            score = 0.0
                    except:
                        score = 0.0
                    
                    # 加密
                    encrypted_blob = PrivacyUtils.encrypt(clean_content, pub_key_pem)
                    
                    packet = {
                        "score": score,
                        "blob": encrypted_blob,
                        "source": self.node_id 
                    }
                    scored_packets.append(packet)
            
            return scored_packets

        except Exception as e:
            print(f"Retrieval Failed: {e}")
            import traceback
            traceback.print_exc()
            return []