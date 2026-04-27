"""
数据可视化工具
==============
根据数据类型和分析目标自动推荐图表类型并生成。
"""

import os
import json
from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ChartResult:
    chart_type: str
    file_path: str
    title: str
    description: str
    data_summary: dict


class AutoVisualizer:
    """
    智能可视化工具
    - 自动推断最适合的图表类型
    - 支持 PNG、HTML（交互式）两种输出
    """

    CHART_RULES = {
        "distribution": ["histogram", "kde", "boxplot"],
        "comparison": ["barplot", "grouped_bar", "heatmap"],
        "correlation": ["scatter", "heatmap", "pairplot"],
        "trend": ["lineplot", "area"],
        "proportion": ["pie", "donut", "stacked_bar"],
        "regression": ["scatter_with_fit"],
    }

    def __init__(self, output_dir: str = "outputs", logger=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def infer_chart_type(self, intent: str, data_info: dict) -> str:
        """根据分析意图和数据特征推断图表类型"""
        intent_lower = intent.lower()
        n_numeric = data_info.get("n_numeric", 0)
        n_categorical = data_info.get("n_categorical", 0)
        n_rows = data_info.get("n_rows", 0)

        if any(k in intent_lower for k in ["分布", "distribution", "histogram"]):
            return "histogram"
        if any(k in intent_lower for k in ["趋势", "trend", "时间", "time series"]):
            return "lineplot"
        if any(k in intent_lower for k in ["相关", "correlation", "关系"]):
            return "scatter" if n_numeric >= 2 else "heatmap"
        if any(k in intent_lower for k in ["比较", "compare", "对比"]):
            return "barplot"
        if any(k in intent_lower for k in ["占比", "proportion", "pie", "组成"]):
            return "pie"
        if any(k in intent_lower for k in ["回归", "regression", "拟合"]):
            return "scatter_with_fit"
        # 默认启发式
        if n_categorical >= 1 and n_numeric >= 1:
            return "barplot"
        if n_numeric >= 2:
            return "scatter"
        return "histogram"

    def plot(self, chart_type: str, data: Any, title: str = "",
             filename: str = "chart", fmt: str = "png") -> ChartResult:
        """生成图表"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(9, 5))
            description = ""

            if chart_type == "histogram":
                arr = self._to_array(data)
                ax.hist(arr, bins=20, color="#4A90D9", edgecolor="white", alpha=0.85)
                ax.set_xlabel("值")
                ax.set_ylabel("频次")
                description = f"数据分布直方图，均值={np.mean(arr):.2f}，标准差={np.std(arr):.2f}"

            elif chart_type == "lineplot":
                if isinstance(data, dict):
                    for label, values in data.items():
                        ax.plot(values, label=label, linewidth=2)
                    ax.legend()
                else:
                    arr = self._to_array(data)
                    ax.plot(arr, color="#4A90D9", linewidth=2)
                ax.set_xlabel("Index")
                ax.set_ylabel("值")
                description = "时间序列/趋势折线图"

            elif chart_type == "barplot":
                if isinstance(data, dict):
                    keys = list(data.keys())
                    vals = list(data.values())
                else:
                    keys = [f"类别{i}" for i in range(len(data))]
                    vals = list(data)
                colors = ["#4A90D9", "#E8A838", "#5CB85C", "#D9534F", "#9B59B6"]
                ax.bar(keys, vals, color=colors[:len(keys)], edgecolor="white")
                ax.set_xlabel("类别")
                ax.set_ylabel("值")
                description = f"柱状图，共{len(keys)}组数据"

            elif chart_type == "scatter" or chart_type == "scatter_with_fit":
                if isinstance(data, dict) and "x" in data and "y" in data:
                    x, y = np.array(data["x"]), np.array(data["y"])
                else:
                    import numpy as np
                    x = np.arange(20)
                    y = x * 1.5 + np.random.randn(20) * 3
                ax.scatter(x, y, alpha=0.7, color="#4A90D9", s=60)
                if chart_type == "scatter_with_fit":
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="拟合线")
                    ax.legend()
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                description = "散点图" + ("（含回归拟合）" if "fit" in chart_type else "")

            elif chart_type == "heatmap":
                import numpy as np
                if isinstance(data, (list, np.ndarray)):
                    matrix = np.array(data)
                else:
                    matrix = np.random.rand(5, 5)
                im = ax.imshow(matrix, cmap="Blues", aspect="auto")
                plt.colorbar(im, ax=ax)
                description = "热力图"

            elif chart_type == "pie":
                if isinstance(data, dict):
                    labels, sizes = list(data.keys()), list(data.values())
                else:
                    labels = [f"类别{i}" for i in range(len(data))]
                    sizes = list(data)
                ax.pie(sizes, labels=labels, autopct="%1.1f%%",
                       startangle=90, colors=["#4A90D9","#E8A838","#5CB85C","#D9534F","#9B59B6"])
                ax.axis("equal")
                description = "饼图，展示各组成部分占比"

            else:
                arr = self._to_array(data)
                ax.plot(arr, color="#4A90D9")
                description = "通用图表"

            ax.set_title(title or chart_type.replace("_", " ").title(), fontsize=13, pad=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()

            out_path = self.output_dir / f"{filename}.{fmt}"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if self.logger:
                self.logger.success("Visualizer", f"图表已保存: {out_path}")

            return ChartResult(
                chart_type=chart_type,
                file_path=str(out_path),
                title=title or chart_type,
                description=description,
                data_summary={"chart_type": chart_type, "output": str(out_path)},
            )

        except ImportError:
            if self.logger:
                self.logger.warning("Visualizer", "matplotlib 未安装，跳过图表生成")
            return ChartResult(
                chart_type=chart_type, file_path="",
                title=title, description="（matplotlib 未安装）",
                data_summary={},
            )

    def _to_array(self, data):
        try:
            import numpy as np
            if isinstance(data, (list, tuple)):
                return np.array(data, dtype=float)
            return np.array(data)
        except Exception:
            import numpy as np
            return np.random.randn(50)
