# MindPilot — 多模态智能科研助手 Agent 系统

> 高级人工智能研究生课程 · 六人小组项目

## 项目简介

MindPilot 是一个面向科研全流程的多 Agent 协作系统。用户输入一个科研问题后，系统自动完成：

**文献检索 → 实验设计 → 代码实现 → 数据分析 → 报告生成**

## 系统架构

```
用户请求
   │
   ▼
中央编排器 (Orchestrator)
   │
   ├── 模块① 任务规划 Agent     (ReAct + Tree-of-Thought)
   ├── 模块② 文献检索 Agent     (RAG + ArXiv API + 知识图谱)
   ├── 模块③ 代码生成 Agent     (Code LLM + 沙箱执行 + 自动调试)
   ├── 模块④ 数据分析 Agent     (统计分析 + 自动可视化 + 多格式报告)
   ├── 模块⑤ 多Agent通信框架    (消息队列 + 并发调度 + 日志)
   └── 模块⑥ 评估与反思模块     (LLM-as-Judge + Self-Reflection)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp config.example.env .env
# 编辑 .env，填入你的 OpenAI API Key（或兼容接口）
```

### 3. 运行 Demo

```bash
python examples/demo.py
```

### 4. 交互模式

```bash
python main.py
```

## 目录结构

```
mindpilot/
├── main.py                    # 主入口（交互模式）
├── config.py                  # 全局配置
├── requirements.txt
├── orchestrator/
│   └── orchestrator.py        # 中央编排器
├── agents/
│   ├── planning_agent.py      # 模块① 任务规划
│   ├── literature_agent.py    # 模块② 文献检索
│   ├── code_agent.py          # 模块③ 代码生成
│   ├── analysis_agent.py      # 模块④ 数据分析
│   └── evaluation_agent.py    # 模块⑥ 评估反思
├── framework/
│   ├── communication.py       # 模块⑤ 消息通信协议
│   ├── scheduler.py           # 并发调度器
│   └── logger.py              # 执行日志系统
├── tools/
│   ├── arxiv_search.py        # ArXiv 文献搜索
│   ├── code_executor.py       # 代码沙箱执行
│   ├── visualizer.py          # 数据可视化
│   └── report_generator.py    # 多格式报告生成
├── memory/
│   └── memory_store.py        # 向量记忆存储
├── evaluation/
│   ├── benchmark.py           # 评估基准集
│   └── metrics.py             # 评估指标计算
├── examples/
│   └── demo.py                # 完整演示脚本
└── tests/
    ├── test_planning.py
    ├── test_literature.py
    ├── test_code.py
    ├── test_analysis.py
    └── test_evaluation.py
```

## 六大模块说明

| 模块 | 负责人 | 核心技术 | 评估指标 |
|------|--------|----------|----------|
| ① 任务规划 | 成员① | ReAct, Tree-of-Thought, 记忆 | 任务分解正确率、规划耗时 |
| ② 文献检索 | 成员② | RAG, ArXiv API, 知识图谱 | Recall@5, ROUGE-L |
| ③ 代码生成 | 成员③ | Code LLM, 沙箱, AST分析 | Pass@1, Pass@5 |
| ④ 数据分析 | 成员④ | Pandas, Plotly, 统计检验 | 分析准确率, 多格式输出 |
| ⑤ 通信框架 | 成员⑤ | 消息队列, 并发调度, 日志 | 吞吐量, 集成测试通过率 |
| ⑥ 评估反思 | 成员⑥ | LLM-as-Judge, Self-Reflection | 基准集覆盖, 对比实验 |

## 配置说明

编辑 `config.py` 可修改：
- LLM 模型与 API 地址（支持 OpenAI / 本地 Ollama / 任何兼容接口）
- 最大反思轮数、评分阈值
- 向量存储路径
- 日志级别

## 无 API Key 模式

若未配置 API Key，系统自动切换到 **Mock 模式**，使用模拟 LLM 响应运行完整流程，适合开发调试与演示。
