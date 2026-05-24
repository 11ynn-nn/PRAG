import asyncio
import ollama
import json
from retrieve2gate import EdgeNode_Retrieve # 复用你现有的类
from privacy_utils import PrivacyUtils
from cloud_adapter import BlindCloudRouter
from edge_responder import EdgeResponderService

# === 模拟环境构建 ===
# 初始化云端和几个协作者
CLOUD = BlindCloudRouter("servers_data/leader-a/chroma_db")

# 模拟协作者的知识库 (这里包含真值！)
RESPONDERS = {
    "edge-a2": EdgeResponderService("edge-a2", [
        "Rage is a shooter game by id Software.", 
        "Sierra published many quest games."
    ]),
    "edge-a1": EdgeResponderService("edge-a1", [
        "F.E.A.R. is a psychological horror video game.", 
        "Published by Sierra Entertainment.",
        "The antagonist is Alma Wade, a mysterious girl."
    ]),
    "edge-a3": EdgeResponderService("edge-a3", [
        "Sang-Wook Cheong is a physicist.",
        "Mobile games are popular in Asia."
    ])
}

class SecureEdgeNode(EdgeNode_Retrieve):
    async def infer_securely(self, question):
        # 1. 先尝试本地 (复用你的 retrieve2gate 逻辑)
        # 这里为了演示，假设本地没有，直接触发 gate
        print(f"\n🔒 [{self.node_id}] User Q: \"{question}\"")
        
        # 模拟 Gate 判定：COLLABORATE
        # 实际代码: output = await super().infer_and_judge(question)
        decision = "COLLABORATE" 
        print(f"  ❌ Local Gate: {decision}. Initiating Privacy Protocol.")
        
        if decision == "COLLABORATE":
            return await self.run_privacy_protocol(question)
        
    async def run_privacy_protocol(self, question):
        # --- Step 1: 准备隐私载荷 ---
        # A. 生成 Query 向量
        emb_resp = ollama.embed(model="nomic-embed-text", input=question)
        raw_vec = emb_resp['embeddings'][0]
        
        # B. 加扰动 (Noise)
        noisy_vec = PrivacyUtils.add_noise(raw_vec, noise_level=0.01)
        print(f"  🌪️  Vector Perturbed. Noise injected.")
        
        # C. 生成密钥
        priv, pub_pem = PrivacyUtils.generate_keys()
        
        # --- Step 2: 发送给云端 (模拟网络) ---
        print(f"  ☁️  Sending [NoisyVector + PublicKey] to Cloud Leader...")
        
        # 2.1 云端路由 (基于向量)
        target_domains = [x[0] for x in CLOUD.route_by_vector(noisy_vec, top_k=10)]
        target_domains = [str(d) for d in target_domains] # 清洗
        
        print(f"     [Cloud] Routed to domains: {target_domains}")
        
        # 2.2 云端多播筛选
        selected_edges = CLOUD.multicast_by_vector(noisy_vec, target_domains)
        print(f"[Cloud] Selected Responders: {selected_edges}")
        
        # --- Step 3: 边缘响应 (模拟并行调用) ---
        encrypted_results = []
        for edge_id in selected_edges:
            if edge_id in RESPONDERS:
                # 只有 edge_id 能收到 blind vector，云端看不到内容
                blob = RESPONDERS[edge_id].handle_blind_query(noisy_vec, pub_pem)
                encrypted_results.append(blob)
                
        print(f"  📦 Received {len(encrypted_results)} encrypted blobs from cloud.")
        
        # --- Step 4: 解密与生成 ---
        all_context_text = ""
        for i, blob in enumerate(encrypted_results):
            try:
                # 私钥解密
                decrypted_text = PrivacyUtils.decrypt(blob, priv)
                print(f"     🔓 Decrypted content from responder {i+1}")
                
                # 拼接上下文
                all_context_text += f"\n=== Knowledge from External Device {i+1} ===\n"
                all_context_text += decrypted_text + "\n"
                
            except Exception as e:
                print(f"     ❌ Decrypt failed: {e}")

        if not all_context_text:
            return "Sorry, found nothing securely."

        # 直接把解密出来的文本塞给 LLM
        final_prompt = (
            f"Use the following retrieved context to answer.Keep it concise (under 50 words).\n"
            f"Context:\n{all_context_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        
        print("  🧠 Generating Final Answer locally...")
        ans = await self.rag.llm_model_func(final_prompt)
        return ans

# ================= 运行测试 =================
async def main():
    # 假设这是 edge-a1
    node = SecureEdgeNode("edge-a1", "servers_data/edge-a1")
    await node.initialize()
    
    # 测试那个 F.E.A.R. 问题
    q = "What video game published by Sierra Entertainment includes an antagonist figure who's mystery is the core of the series?"
    
    final_answer = await node.infer_securely(q)
    print(f"\n✅ FINAL ANSWER:\n{final_answer}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())