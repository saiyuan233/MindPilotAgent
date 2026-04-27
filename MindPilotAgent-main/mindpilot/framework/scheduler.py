"""
模块⑤ — 并发调度器
===================
当多个子任务无依赖关系时，触发多 Agent 并行执行。
基于 DAG（有向无环图）的拓扑排序调度。
"""

import time
import asyncio
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class Task:
    """调度单元"""
    task_id: str
    agent_name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)   # 前置任务 ID 列表
    priority: int = 0          # 数字越大优先级越高
    timeout: Optional[float] = None
    result: Any = None
    status: str = "pending"    # pending | running | done | failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 3)
        return None


class DAGScheduler:
    """
    DAG 拓扑排序并发调度器
    - 拓扑排序找出可并行的任务层
    - 同层任务并发执行，跨层有序
    - 单任务失败不阻断独立分支
    """

    def __init__(self, max_concurrent: int = 4, logger=None):
        self.max_concurrent = max_concurrent
        self.logger = logger
        self._tasks: dict[str, Task] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None

    def add_task(self, task: Task):
        self._tasks[task.task_id] = task

    def _topological_layers(self) -> list[list[str]]:
        """将 DAG 分层，同层可并行执行"""
        in_degree = {tid: len(t.depends_on) for tid, t in self._tasks.items()}
        children = defaultdict(list)
        for tid, task in self._tasks.items():
            for dep in task.depends_on:
                children[dep].append(tid)

        layers = []
        ready = deque(tid for tid, deg in in_degree.items() if deg == 0)

        while ready:
            layer = []
            next_ready = deque()
            for tid in ready:
                layer.append(tid)
            layers.append(layer)
            for tid in layer:
                for child in children[tid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            ready = next_ready

        return layers

    async def _run_task(self, task: Task) -> Any:
        async with self._semaphore:
            task.status = "running"
            task.start_time = time.time()
            if self.logger:
                self.logger.info("Scheduler", f"▶ 启动任务 {task.task_id} [{task.agent_name}]")
            try:
                if asyncio.iscoroutinefunction(task.func):
                    if task.timeout:
                        result = await asyncio.wait_for(
                            task.func(*task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                    else:
                        result = await task.func(*task.args, **task.kwargs)
                else:
                    # 同步函数在线程池中运行
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: task.func(*task.args, **task.kwargs)
                    )
                task.result = result
                task.status = "done"
                task.end_time = time.time()
                if self.logger:
                    self.logger.success("Scheduler",
                        f"✓ 任务 {task.task_id} 完成 [{task.duration}s]")
                return result
            except Exception as e:
                task.status = "failed"
                task.end_time = time.time()
                task.result = None
                if self.logger:
                    self.logger.error("Scheduler", f"✗ 任务 {task.task_id} 失败: {e}")
                return None

    async def run_all(self) -> dict[str, Any]:
        """按 DAG 拓扑顺序执行所有任务"""
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        layers = self._topological_layers()
        results: dict[str, Any] = {}

        if self.logger:
            self.logger.info("Scheduler",
                f"DAG 分层完成: {len(layers)} 层，{len(self._tasks)} 个任务")

        for layer_idx, layer in enumerate(layers):
            # 跳过依赖失败的任务
            runnable = []
            for tid in layer:
                task = self._tasks[tid]
                deps_ok = all(
                    self._tasks[dep].status == "done"
                    for dep in task.depends_on
                )
                if deps_ok:
                    runnable.append(task)
                else:
                    task.status = "failed"
                    if self.logger:
                        self.logger.warning("Scheduler",
                            f"跳过 {tid}（前置任务失败）")

            if self.logger and runnable:
                self.logger.info("Scheduler",
                    f"层 {layer_idx+1}/{len(layers)}: 并发执行 {len(runnable)} 个任务")

            # 同层任务并发
            coros = [self._run_task(t) for t in runnable]
            layer_results = await asyncio.gather(*coros, return_exceptions=False)

            for task, result in zip(runnable, layer_results):
                results[task.task_id] = result

        return results

    def get_stats(self) -> dict:
        done = sum(1 for t in self._tasks.values() if t.status == "done")
        failed = sum(1 for t in self._tasks.values() if t.status == "failed")
        durations = [t.duration for t in self._tasks.values() if t.duration]
        return {
            "total": len(self._tasks),
            "done": done,
            "failed": failed,
            "avg_duration_s": round(sum(durations) / len(durations), 3) if durations else 0,
        }


# ── 简化同步版（非 async 场景使用）──────────────────────────────
class SyncScheduler:
    """同步版调度器，按拓扑顺序串行执行（用于非 async 场景）"""

    def __init__(self, logger=None):
        self.logger = logger
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task):
        self._tasks[task.task_id] = task

    def run_all(self) -> dict[str, Any]:
        # 拓扑排序
        visited, order = set(), []
        def dfs(tid):
            if tid in visited:
                return
            visited.add(tid)
            for dep in self._tasks[tid].depends_on:
                if dep in self._tasks:
                    dfs(dep)
            order.append(tid)

        for tid in self._tasks:
            dfs(tid)

        results = {}
        for tid in order:
            task = self._tasks[tid]
            deps_ok = all(
                self._tasks[dep].status == "done"
                for dep in task.depends_on
                if dep in self._tasks
            )
            if not deps_ok:
                task.status = "failed"
                continue

            task.status = "running"
            task.start_time = time.time()
            try:
                task.result = task.func(*task.args, **task.kwargs)
                task.status = "done"
            except Exception as e:
                task.status = "failed"
                if self.logger:
                    self.logger.error("Scheduler", f"任务 {tid} 失败: {e}")
            task.end_time = time.time()
            results[tid] = task.result

        return results
