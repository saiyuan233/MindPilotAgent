"""
模块② — 文献检索与知识图谱 Agent
===================================
混合检索（关键词 + 语义向量）+ 知识图谱构建 + 结构化摘要生成。
已升级：引入 Cross-Encoder 深度交互模型进行高精度精排 (Rerank)。
已升级：知识图谱持久化 + 语义节点扩展 + 图增强检索。
"""

import os
import json
import re
import math
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    node_id: str
    node_type: str          # paper | author | method | category | keyword (新增 keyword)
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """知识图谱边"""
    source: str
    target: str
    relation: str           # cites | uses_method | belongs_to | authored_by | has_keyword
    weight: float = 1.0


class LightKnowledgeGraph:
    """
    轻量持久化知识图谱（基于纯 Python，支持存盘与图增强检索）
    """

    def __init__(self, storage_path: str = "memory/store/kg.json"):
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[KnowledgeEdge] = {}
        self.edges: list[KnowledgeEdge] = []
        self._adj: dict[str, list[str]] = {}   # 邻接表（用于多跳推理）
        self.storage_path = storage_path
        
        # 启动时自动加载本地历史图谱
        self.load_from_disk()

    def add_node(self, node: KnowledgeNode):
        self.nodes[node.node_id] = node
        self._adj.setdefault(node.node_id, [])

    def add_edge(self, edge: KnowledgeEdge):
        self.edges.append(edge)
        self._adj.setdefault(edge.source, []).append(edge.target)
        self._adj.setdefault(edge.target, []).append(edge.source)

    def add_paper(self, paper) -> str:
        """从 Paper 对象构建图谱节点与关系，新增结构化摘要与关键词"""
        pid = f"paper:{paper.arxiv_id}"
        
        # 1. 论文节点（在 properties 中存入结构化摘要）
        self.add_node(KnowledgeNode(
            node_id=pid, node_type="paper",
            label=paper.title[:100],
            properties={
                "year": paper.published[:4], 
                "url": paper.url,
                "relevance": paper.relevance_score,
                "summary": paper.structured_summary  # 【新增】：将摘要完整入库
            }
        ))
        
        # 2. 作者节点
        for author in paper.authors[:3]:
            aid = f"author:{author.replace(' ', '_')}"
            self.add_node(KnowledgeNode(node_id=aid, node_type="author", label=author))
            self.add_edge(KnowledgeEdge(pid, aid, "authored_by"))
            
        # 3. 类别节点
        for cat in paper.categories[:2]:
            cid = f"cat:{cat}"
            self.add_node(KnowledgeNode(node_id=cid, node_type="category", label=cat))
            self.add_edge(KnowledgeEdge(pid, cid, "belongs_to"))
            
        # 4. 关键词节点（【新增】：从类别和标题中简单提取关键词建点）
        keywords = paper.categories + [w for w in paper.title.split() if len(w) > 5]
        for kw in keywords[:5]:
            kid = f"kw:{kw.lower()}"
            self.add_node(KnowledgeNode(node_id=kid, node_type="keyword", label=kw))
            self.add_edge(KnowledgeEdge(pid, kid, "has_keyword"))
            
        return pid

    def search_relevant_papers(self, query: str) -> list[str]:
        """【新增】：从本地图谱中基于关键词检索相关的历史论文ID"""
        query_words = set(query.lower().split())
        relevant_pids = []
        
        for nid, node in self.nodes.items():
            if node.node_type == "paper":
                # 拼接标题和结构化摘要作为检索文本
                node_text = node.label.lower()
                summary = node.properties.get("summary")
                if isinstance(summary, dict):
                    node_text += " " + " ".join(str(v).lower() for v in summary.values())
                
                # 如果查询词在这个节点的文本中出现，则视为匹配
                if any(word in node_text for word in query_words):
                    relevant_pids.append(nid)
                    
        return relevant_pids

    def save_to_disk(self):
        """【新增】：将当前图谱状态持久化为 JSON 文件"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges]
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self):
        """【新增】：从 JSON 文件加载历史图谱"""
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for nid, n_data in data.get("nodes", {}).items():
                    self.nodes[nid] = KnowledgeNode(**n_data)
                    self._adj.setdefault(nid, [])
                for e_data in data.get("edges", []):
                    edge = KnowledgeEdge(**e_data)
                    self.edges.append(edge)
                    self._adj.setdefault(edge.source, []).append(edge.target)
                    self._adj.setdefault(edge.target, []).append(edge.source)
        except Exception:
            pass

    def multi_hop_query(self, start_label: str, hops: int = 2) -> list[KnowledgeNode]:
        """从起点出发进行 N 跳检索"""
        start_nodes = [
            nid for nid, n in self.nodes.items()
            if start_label.lower() in n.label.lower()
        ]
        if not start_nodes:
            return []
        visited = set(start_nodes)
        frontier = set(start_nodes)
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                for neighbor in self._adj.get(nid, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
        return [self.nodes[nid] for nid in visited if nid in self.nodes]

    def stats(self) -> dict:
        type_counts = {}
        for n in self.nodes.values():
            type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "types": type_counts}


class StructuredSummarizer:
    """文献结构化摘要生成器"""

    def __init__(self, llm_client, max_len: int = 300, logger=None):
        self.llm = llm_client
        self.max_len = max_len
        self.logger = logger

    def summarize(self, paper) -> dict:
        system = (
            "你是学术论文分析专家。请将以下论文摘要压缩为结构化摘要。"
            "以 JSON 格式输出（字段：method, conclusion, limitation），每项不超过60字。"
        )
        text = f"标题：{paper.title}\n摘要：{paper.abstract[:800]}"
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ])
        try:
            m = re.search(r"\{[\s\S]+\}", resp)
            summary = json.loads(m.group(0) if m else resp)
            return {
                "method": summary.get("method", "未提取到"),
                "conclusion": summary.get("conclusion", "未提取到"),
                "limitation": summary.get("limitation", "未提取到"),
            }
        except Exception:
            sents = paper.abstract.split(". ")
            return {
                "method": sents[0][:100] if len(sents) > 0 else "",
                "conclusion": sents[1][:100] if len(sents) > 1 else "",
                "limitation": sents[-1][:100] if len(sents) > 2 else "",
            }


class LiteratureAgent:
    """
    模块② — 文献检索与知识图谱 Agent
    混合检索 + 图增强检索 + 图谱构建与持久化 + 结构化摘要 + Cross-Encoder 深度精排
    """

    AGENT_NAME = "LiteratureAgent"

    def __init__(self, config, llm_client, arxiv_tool, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.arxiv = arxiv_tool
        self.memory = memory_store
        self.logger = logger
        self.summarizer = StructuredSummarizer(llm_client, config.literature.summary_max_len, logger)
        
        # 兼容 config 中可能配置的不同存储路径
        storage_dir = getattr(config, 'memory_dir', 'memory')
        kg_path = os.path.join(storage_dir, "store", "kg.json")
        self.kg = LightKnowledgeGraph(storage_path=kg_path)
        
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker = None 

    def _init_reranker(self):
        if self.reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self.logger.info(self.AGENT_NAME, f"正在加载 Cross-Encoder 模型 ({self.reranker_model_name})...")
            self.reranker = CrossEncoder(self.reranker_model_name, max_length=512)
            self.logger.success(self.AGENT_NAME, "Cross-Encoder 模型加载完成！")
        except ImportError:
            self.logger.warning(self.AGENT_NAME, "未安装 sentence-transformers，将降级使用 TF-IDF 重排序。")
            self.reranker = "fallback"
        except Exception as e:
            self.logger.warning(self.AGENT_NAME, f"Cross-Encoder 模型加载失败，降级使用 TF-IDF: {e}")
            self.reranker = "fallback"

    def run(self, task_description: str, query: str = "") -> dict:
        # search_query = task_description or query
        search_query = query or task_description
        call = self.logger.start_call(self.AGENT_NAME, "literature_search", search_query)

        try:
            # Step 1: 图增强检索 —— 优先探查本地知识图谱中的已有文献
            self.logger.info(self.AGENT_NAME, "正在进行图增强检索（扫描本地知识图谱）...")
            local_pids = self.kg.search_relevant_papers(search_query)
            if local_pids:
                self.logger.info(self.AGENT_NAME, f"💡 从本地记忆中关联到 {len(local_pids)} 篇历史文献。")

            # Step 2: 外部 ArXiv 检索
            self.logger.info(self.AGENT_NAME, "开始 ArXiv 混合检索获取最新前沿...")
            papers = self.arxiv.search(
                search_query,
                max_results=self.config.literature.arxiv_max_results
            )

            # Step 3: 深度语义重排序 (Cross-Encoder)
            if papers:
                papers = self._rerank(papers, search_query)

            # Step 4: 生成结构化摘要 + 更新知识图谱
            self.logger.info(self.AGENT_NAME, f"为 {len(papers)} 篇论文生成摘要并构建图谱关系...")
            for paper in papers:
                # 只有当还没生成摘要时才生成
                if not paper.structured_summary:
                    paper.structured_summary = self.summarizer.summarize(paper)
                # 无论是否已有，加入图谱（会自动建立 keyword, summary 等新属性）
                self.kg.add_paper(paper)

            # 【重要】 Step 5: 图谱持久化落盘
            self.kg.save_to_disk()
            self.logger.info(self.AGENT_NAME, "知识图谱状态已持久化。")

            # Step 6: 评估指标与综述生成
            recall_5 = self._compute_recall_at_k(papers, k=5)
            recall_10 = self._compute_recall_at_k(papers, k=10)
            review = self._generate_review(search_query, papers[:5])

            # Step 7: 存入短期记忆流
            self.memory.add(
                content=f"文献检索: {search_query}，找到 {len(papers)} 篇",
                agent=self.AGENT_NAME,
                payload={"papers": [p.to_dict() for p in papers[:5]]},
                tags=["literature"],
            )

            result = {
                "papers": [p.to_dict() for p in papers],
                "top_papers": [p.to_dict() for p in papers[:self.config.literature.retrieval_top_k]],
                "knowledge_graph": self.kg.stats(),
                "literature_review": review,
                "metrics": {"recall@5": recall_5, "recall@10": recall_10},
                "total_found": len(papers),
                "local_kg_hits": len(local_pids)  # 新增返回字段
            }
            self.logger.finish_call(call, result)
            self._print_results(papers[:5])
            return result

        except Exception as e:
            self.logger.fail_call(call, str(e))
            raise

    def _rerank(self, papers, query: str):
        self._init_reranker()
        if self.reranker == "fallback":
            return self._fallback_rerank(papers, query)

        self.logger.info(self.AGENT_NAME, f"正在使用 Cross-Encoder 对 {len(papers)} 篇候选论文进行深度重排...")
        pairs = [[query, p.title + " " + p.abstract] for p in papers]

        try:
            scores = self.reranker.predict(pairs)
            for i, p in enumerate(papers):
                sigmoid_score = 1 / (1 + math.exp(-scores[i]))
                p.relevance_score = round(float(sigmoid_score), 3)

            sorted_papers = sorted(papers, key=lambda p: p.relevance_score, reverse=True)
            self.logger.success(self.AGENT_NAME, "重排序完成。")
            return sorted_papers
        except Exception as e:
            self.logger.warning(self.AGENT_NAME, f"Cross-Encoder 推理异常，降级回 TF-IDF: {e}")
            return self._fallback_rerank(papers, query)

    def _fallback_rerank(self, papers, query: str):
        query_words = set(query.lower().split())
        for p in papers:
            text = (p.title + " " + p.abstract).lower()
            words = text.split()
            total = len(words)
            if total == 0: continue
            tf_score = sum(text.count(w) / total for w in query_words)
            p.relevance_score = round(0.6 * p.relevance_score + 0.4 * min(tf_score * 10, 1.0), 3)
        return sorted(papers, key=lambda p: p.relevance_score, reverse=True)

    def _compute_recall_at_k(self, papers, k: int) -> float:
        top_k = papers[:k]
        relevant = sum(1 for p in top_k if p.relevance_score > 0.3)
        total_relevant = sum(1 for p in papers if p.relevance_score > 0.3)
        if total_relevant == 0: return 0.0
        return round(relevant / total_relevant, 3)

    def _generate_review(self, query: str, papers: list) -> str:
        if not papers: return "未找到相关文献。"
        paper_summaries = "\n".join([
            f"[{i+1}] {p.title}\n   方法：{p.structured_summary.get('method','') if p.structured_summary else ''}"
            for i, p in enumerate(papers)
        ])
        system = "你是学术写作专家。请根据以下论文列表，写一段200字以内的中文文献综述。"
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"研究主题：{query}\n\n相关论文：\n{paper_summaries}"}
        ])
        return resp[:600]

    def _print_results(self, papers: list):
        print(f"\n{'━'*58}")
        print(f"  📚 文献检索结果 (Top {len(papers)})")
        print(f"{'━'*58}")
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
            print(f"  [{i}] {p.title[:52]}")
            print(f"       {authors} ({p.published[:4]}) | 深度语义得分: {p.relevance_score:.2f}")
        print(f"  知识图谱: {self.kg.stats()}")
        print(f"{'━'*58}\n")