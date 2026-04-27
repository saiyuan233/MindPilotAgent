"""
评估基准集与指标计算
====================
20 条科研任务 + 关键词参考答案，用于横向对比实验。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkCase:
    """单条基准测试用例"""
    case_id: str
    question: str
    category: str              # literature | code | analysis | mixed
    reference_keywords: list[str]
    difficulty: str = "medium" # easy | medium | hard
    expected_length: str = "medium"


# 20 条基准用例（涵盖文献、代码、分析三类）
BENCHMARK_CASES = [
    # ── 文献类 ──
    BenchmarkCase("L01", "Transformer 的自注意力机制是如何工作的？请解释 Q、K、V 矩阵的作用。",
                  "literature", ["Query", "Key", "Value", "点积", "softmax", "注意力权重"], "medium"),
    BenchmarkCase("L02", "BERT 和 GPT 的预训练目标有何根本区别？",
                  "literature", ["MLM", "NSP", "自回归", "双向", "单向", "掩码"], "medium"),
    BenchmarkCase("L03", "联邦学习的核心优势和主要挑战是什么？",
                  "literature", ["隐私", "数据不出本地", "通信", "异构", "FedAvg"], "easy"),
    BenchmarkCase("L04", "扩散模型（Diffusion Model）的前向和反向过程分别是什么？",
                  "literature", ["马尔可夫", "噪声", "去噪", "DDPM", "生成"], "hard"),
    BenchmarkCase("L05", "对比学习的核心思想是什么？列举两个代表性方法。",
                  "literature", ["正样本", "负样本", "SimCLR", "MoCo", "InfoNCE"], "medium"),
    BenchmarkCase("L06", "知识蒸馏是如何将大模型压缩为小模型的？",
                  "literature", ["教师", "学生", "软标签", "温度", "KL散度"], "medium"),
    BenchmarkCase("L07", "图神经网络（GNN）的消息传递机制是什么？",
                  "literature", ["邻居聚合", "消息", "更新", "过平滑", "GCN"], "medium"),
    # ── 代码类 ──
    BenchmarkCase("C01", "用 Python 实现一个简单的线性回归，并打印 R² 分数。",
                  "code", ["LinearRegression", "fit", "r2_score", "predict", "numpy"], "easy"),
    BenchmarkCase("C02", "用 Python 实现 K-means 聚类并可视化聚类结果。",
                  "code", ["KMeans", "fit", "cluster_centers_", "scatter", "matplotlib"], "medium"),
    BenchmarkCase("C03", "用 Python 计算两个向量的余弦相似度，不使用外部库。",
                  "code", ["dot", "norm", "sum", "sqrt", "cos"], "easy"),
    BenchmarkCase("C04", "用 Pandas 对一个 CSV 数据集进行描述性统计分析。",
                  "code", ["read_csv", "describe", "info", "isnull", "value_counts"], "easy"),
    BenchmarkCase("C05", "用 Python 实现梯度下降算法求函数 f(x)=x²+3x+2 的最小值。",
                  "code", ["gradient", "learning_rate", "iteration", "update", "derivative"], "medium"),
    BenchmarkCase("C06", "用 sklearn 训练一个随机森林分类器并输出特征重要性。",
                  "code", ["RandomForestClassifier", "feature_importances_", "fit", "score"], "medium"),
    # ── 分析类 ──
    BenchmarkCase("A01", "对一组正态分布数据进行假设检验，判断均值是否等于 0。",
                  "analysis", ["t检验", "p值", "显著性", "零假设", "拒绝"], "medium"),
    BenchmarkCase("A02", "分析两个变量之间的 Pearson 相关系数并解释结果。",
                  "analysis", ["Pearson", "相关系数", "p值", "线性", "散点图"], "easy"),
    BenchmarkCase("A03", "对一个数据集执行完整的 EDA，包括缺失值、分布和异常值分析。",
                  "analysis", ["缺失值", "分布", "箱线图", "异常值", "describe"], "medium"),
    BenchmarkCase("A04", "对三组数据进行单因素 ANOVA 检验并解释 F 统计量。",
                  "analysis", ["ANOVA", "F统计量", "p值", "组间", "组内"], "hard"),
    # ── 综合类 ──
    BenchmarkCase("M01", "研究 BERT 在文本分类任务上的性能，包括文献调研和代码实现。",
                  "mixed", ["BERT", "fine-tuning", "文本分类", "accuracy", "Huggingface"], "hard"),
    BenchmarkCase("M02", "对比 SGD 和 Adam 优化器在神经网络训练中的收敛速度，附可视化。",
                  "mixed", ["SGD", "Adam", "学习率", "收敛", "损失曲线"], "hard"),
    BenchmarkCase("M03", "调研并实现一个简单的 Attention 机制，可视化注意力权重。",
                  "mixed", ["注意力", "权重", "softmax", "热力图", "可视化"], "hard"),
]


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def keyword_recall(output: str, keywords: list[str]) -> float:
        """关键词召回率"""
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw.lower() in output.lower())
        return round(hits / len(keywords), 3)

    @staticmethod
    def recall_at_k(retrieved: list, relevant_keywords: list[str], k: int) -> float:
        """Recall@K：Top-K 结果中覆盖的关键词比例"""
        top_k_text = " ".join(str(r) for r in retrieved[:k])
        return MetricsCalculator.keyword_recall(top_k_text, relevant_keywords)

    @staticmethod
    def rouge_l(hypothesis: str, reference: str) -> float:
        """ROUGE-L F1 近似计算"""
        def lcs_length(a, b):
            m, n = len(a), len(b)
            if m > 200: a = a[:200]; m = 200
            if n > 200: b = b[:200]; n = 200
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        h = hypothesis.lower().split()
        r = reference.lower().split()
        if not h or not r:
            return 0.0
        lcs = lcs_length(h, r)
        p = lcs / len(h)
        rc = lcs / len(r)
        return round(2 * p * rc / (p + rc), 4) if (p + rc) > 0 else 0.0

    @staticmethod
    def pass_at_k(results: list[bool], k: int) -> float:
        """Pass@K：K 次尝试中至少一次通过的概率"""
        n = len(results)
        c = sum(results)
        if n == 0 or k > n:
            return 0.0
        if c == 0:
            return 0.0
        # Pass@k = 1 - C(n-c, k) / C(n, k)
        from math import comb
        k = min(k, n)
        if c == 0:
            return 0.0
        # Numerical stable: 1 - prod((n-c-i)/(n-i) for i in range(k))
        prob_none_correct = 1.0
        for i in range(k):
            if n - i > 0:
                prob_none_correct *= max(0, (n - c - i)) / (n - i)
        return round(1.0 - prob_none_correct, 3)
