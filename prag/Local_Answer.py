import asyncio
import os
import json
import numpy as np
import ollama
import chromadb
from collections import defaultdict 
from datetime import datetime
import time
import statistics
from openai import OpenAI 
from nano_graphrag import GraphRAG, QueryParam as NanoQueryParam
from nano_graphrag.base import EmbeddingFunc
import re
import torch
from FlagEmbedding import FlagReranker
import pandas as pd # 用于生成漂亮的报表


class CollaborativeAnswerGenerator:
    def __init__(self, model_name="qwen/Qwen2.5-7B-Instruct"):
        
        self.model = model_name
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

        try:
            model_path = "./rerank_model" 
            self.reranker = FlagReranker(model_path, use_fp16=torch.cuda.is_available())
            self.use_reranker = True
            print("Reranker 加载成功！")
        except Exception as e:
            print(f"Reranker 加载失败: {e}")
            self.use_reranker = False

    def generate(self, query, context_map):
        
        print(f"\n>>> [Step 5] 本地最终生成 (via vLLM)")
        all_sources = []
        all_relations = []
        for dev_id, content in context_map.items():
            if content:
                all_sources.extend(self._split_sources(content))
                all_relations.extend(self._extract_relations(content))
        
        if not all_sources and not all_relations:
            # 确保 monitor 在你的全局环境里存在，否则这里会报错
            # monitor.end_phase("Generation") 
            return "I cannot answer this question based on the current knowledge base."

        print(f"   |-- 收集到 {len(all_sources)} 个文本单元, {len(all_relations)} 条关系数据")

        top_clean_sources = []
        if all_sources:
            if self.use_reranker:
                pairs = [[query, u] for u in all_sources]
                scores = self.reranker.compute_score(pairs)
                if isinstance(scores, float): scores = [scores]
                scored_units = list(zip(scores, all_sources))
                scored_units.sort(key=lambda x: x[0], reverse=True)
                top_raw_sources = [item[1] for item in scored_units[:3]]
                print(f"   |-- Source Rerank Top 1 Score: {scored_units[0][0]:.4f}")
            else:
                top_raw_sources = all_sources[:3]
            top_clean_sources = [self._clean_unit(u) for u in top_raw_sources]

        top_relations = []
        if all_relations:
            if self.use_reranker:
                pairs = [[query, r] for r in all_relations]
                scores = self.reranker.compute_score(pairs)
                if isinstance(scores, float): scores = [scores]
                scored_rels = list(zip(scores, all_relations))
                scored_rels.sort(key=lambda x: x[0], reverse=True)
                top_relations = [item[1] for item in scored_rels[:30]]
                print(f"   |-- Relation Rerank Top 1 Score: {scored_rels[0][0]:.4f}")
            else:
                top_relations = all_relations
            top_clean_relations = [self._clean_unit(r) for r in top_relations]

        # 3. 组装最终 Context 字符串
        relations_text = "\n".join(top_clean_relations) if top_clean_relations else "No specific relations found."
        sources_text = "\n\n".join(top_clean_sources) if top_clean_sources else "No specific text details found."
        
        final_context = f"""
=== Relations ===
{relations_text}

===Context===
{sources_text}
"""
        print(f"--- Refined Context ---\n{final_context}\n-------------------------------")
        
        sys_prompt = """---Role---
You are an expert AI assistant specializing in synthesizing information from provided knowledge. 
Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.
---Instructions---
1. **NOISE FILTERING**: IGNORE all text that does not explicitly match the entities in the User Query.
2. **ANCHORING STRATEGY**: Extract the attribute asked for ONLY from the sentence containing the Subject.
3. **ENTITY DISTINCTION**: Pay close attention to specific subtitles or full names.
4. **Step-by-Step Reasoning**: Analyze Intent -> Extract Facts.
5. **Content & Grounding**: Answer ONLY based on the provided Context.
6. **Formatting**: Keep the answer concise and direct.
"""
        user_prompt = f"""
---Context---
{final_context}

---User Query---
{query}

---Instruction---
Based ONLY on the Context Data above, answer the query. 
If the query mentions a specific subtitle, ensure you find the facts for THAT specific version, not others.

---Answer---
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # RAG 场景推荐 0.0，避免幻觉
                max_tokens=512   # 限制输出长度，保持 Concise
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"vLLM API 调用失败: {e}")
            return "Error during generation."

    async def _evaluate_confidence(self, question: str, response: str) -> bool:
        """
        置信度评估器
        返回: True (自信/包含答案), False (不确定/拒答)
        """
        # === 1. 关键词快筛 (精简版) ===
        negative_signals = [
            "general knowledge", "common knowledge", # 偷跑信号
            "not mentioned", "no mention",           # 拒答信号
            "no information", "insufficient information",
            "couldn't find", "cannot find",
            "does not provide", "doesn't provide",
            "outside of the context", "training data",
            "i cannot answer"
        ]
        
        response_lower = str(response).lower()
        for kw in negative_signals:
            if kw in response_lower:
                print(f"   [Gate Fast-Fail] Detected negative signal: '{kw}'")
                return False

        # === 2. LLM 深度裁判 (Deep Judge) ===
        judge_prompt = (
            f"Task: Check if the AI response successfully answers the query using the provided context.\n\n"
            f"Query: \"{question}\"\n"
            f"Response: \"{response}\"\n\n"
            f"Evaluation Rules:\n"
            f"1. If the response says it CANNOT answer or the info is missing -> Output 'UNCERTAIN'.\n"
            f"2. If the response admits using 'general knowledge' or 'common knowledge' instead of the context -> Output 'UNCERTAIN'.\n"
            f"3. If the response provides an answer (even if it infers it from related facts in the context) -> Output 'CONFIDENT'.\n"
            f"4. Do NOT punish the model for saying 'Based on the provided context'. That is good.\n\n"
            f"Verdict (CONFIDENT / UNCERTAIN):"
        )

        try:
            # 适配: 使用类内部的 self.client 替代 ollama_model_complete
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0, # 温度设为 0 以保持冷静
                max_tokens=10
            )
            judgment = completion.choices[0].message.content.strip().upper()
            
            print(f"[Gate Judge] Verdict: {judgment}")
            
            if "UNCERTAIN" in judgment:
                return False
            return True
            
        except Exception as e:
            print(f"   [Gate Error] {e}")
            return True # 默认放行，避免死循环