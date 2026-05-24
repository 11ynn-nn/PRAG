import asyncio
import ollama
from privacy_utils import PrivacyGuard

# 模拟网络发送
from cloud_router import CloudBlindRouter

class EdgeRequester:
    def __init__(self, node_id):
        self.node_id = node_id
        # 生成一次性会话密钥
        self.priv_key, self.pub_key = PrivacyGuard.generate_key_pair()

    async def initiate_secure_query(self, query_text):
        print(f"\n🔒 [{self.node_id}] Initiating Privacy-Preserving Query...")
        
        # 1. 本地 Embedding
        # 这里必须用和大家一致的模型，比如 nomic-embed-text
        embedding_resp = ollama.embed(model="nomic-embed-text", input=query_text)
        original_vec = embedding_resp['embeddings'][0]
        
        # 2. [关键] 注入噪声 (Noise Injection)
        # 0.01 的噪声既能保持大概方向(Computer Science)，又能模糊具体细节(F.E.A.R.)
        noisy_vec = PrivacyGuard.add_noise_to_vector(original_vec, noise_level=0.01)
        
        print(f"   Generating Noisy Vector (Noise Level: 0.01)...")
        print(f"   Original Head: {original_vec[:3]}")
        print(f"   Noisy Head:    {noisy_vec[:3]}")
        
        # 3. 发送给云端 (只发噪声向量 + 公钥)
        # 注意：绝对不发 query_text
        payload = {
            "requester_id": self.node_id,
            "query_vector": noisy_vec, # <--- 只有向量
            "public_key": self.pub_key # <--- 用于回传加密数据
        }
        
        # --- 模拟网络调用云端 ---
        router = CloudBlindRouter()
        encrypted_responses = await router.handle_request(payload)
        
        # 4. 解密并生成
        await self.finalize_answer(query_text, encrypted_responses)

    async def finalize_answer(self, query, encrypted_responses):
        print(f"\n🔓 [{self.node_id}] Received {len(encrypted_responses)} encrypted blobs. Decrypting...")
        
        context_texts = []
        for resp in encrypted_responses:
            try:
                # 使用私钥解密
                json_str = PrivacyGuard.decrypt_data(resp['blob'], self.priv_key)
                # 只有 edge-a1 能看到这段明文！云端看不到！
                print(f"   -> Decrypted from {resp['source']}: {json_str[:60]}...")
                context_texts.append(json_str)
            except Exception as e:
                print(f"   ❌ Decryption failed: {e}")

        # 5. 本地最终生成 (Local Generation)
        # 使用原始 Query + 解密后的文档
        final_prompt = (
            f"Context:\n" + "\n".join(context_texts) + "\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        print(f"   🤖 Running Local LLM with retrieved context...")
        # res = ollama.chat(...) 
        # print("   Final Answer: ...")