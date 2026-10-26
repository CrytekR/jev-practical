"""Jev 类模型放得进去的业务流程。每个场景绑定原语，以及部署时要单独验证的优化。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    state: str
    questions: str
    why: str
    optimizations: tuple[str, ...]
    hold_in_code: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="ticket-routing",
        title="工单分流",
        state="用户来信或工单正文",
        questions="choice 归属团队；score 紧急程度；noul 是否包含账号凭据",
        why="候选团队和紧急等级都能事先写完，每条工单都要判断，输出直接驱动队列。",
        optimizations=("兜底团队", "分差与集中度一起看", "按误分诊成本设接受阈值"),
        hold_in_code="权限、值班表和真正的派单动作",
    ),
    Scenario(
        id="injection-gate",
        title="注入与策略闸门",
        state="即将进入工具或检索的文本",
        questions="noul：是否要求忽略既有指令、套取凭据或改变权限",
        why="热路径上的是非判断，概率可以直接和误拦、漏拦成本比较。",
        optimizations=("在本任务标注上做 Platt 或温度校准", "阈值来自两类错误成本", "低决定性样本转人工"),
        hold_in_code="工具白名单、鉴权和速率限制",
    ),
    Scenario(
        id="rag-filter",
        title="检索结果筛选",
        state="查询与候选片段",
        questions="score 相关程度，或 choice 保留 / 丢弃 / 证据不足",
        why="片段数量多，判断本身不需要生成摘要。",
        optimizations=("同一查询的多个片段共享前缀", "相关度用有序等级而不是无序标签", "证据不足单独成档"),
        hold_in_code="召回、去重和引用拼接",
    ),
    Scenario(
        id="output-check",
        title="生成结果质检",
        state="任务说明加上待检查的回答",
        questions="noul 是否切题；noul 是否触犯给定政策；score 严重程度",
        why="质检问的是封闭属性。严重程度和是否违规是两件独立的事。",
        optimizations=("原子问题并行问", "非线性损失用期望损失而不是期望等级", "质检器与被检查模型分开"),
        hold_in_code="拦截后的改写、重试次数和发布开关",
    ),
    Scenario(
        id="agent-action",
        title="文本状态上的下一步动作",
        state="程序已经抽好的页面文本、可行动作和约束摘要",
        questions="choice 下一步动作；noul 证据是否够自动执行",
        why="动作表是调用时给出的，决策模型只负责在表内选择。",
        optimizations=("单代号读出以压住长动作名的成本", "打乱代号测量位置偏置", "返回时核对状态版本"),
        hold_in_code="点击时序、碰撞和过期指令的丢弃",
    ),
    Scenario(
        id="eval-loop",
        title="优化循环里的评审",
        state="一次运行的轨迹或回答",
        questions="score 评分量规；noul 是否通过某条检查",
        why="评审会被调用很多次，输入是同一类封闭量规。",
        optimizations=("批量问题共享轨迹前缀", "报告 ECE 与覆盖率而不只报准确率", "标注集与训练轨迹分开"),
        hold_in_code="不可由评审模型兼任的最终验收",
    ),
)


POOR_FITS: tuple[str, ...] = (
    "需要写出给人读的正文、代码或逐步推导",
    "答案无法事先收成选项或等级",
    "后一个问题必须看见前一个问题的选择才能定义候选",
    "图像、时序控制和权限判断被交给了本应留在程序里的部分",
)
