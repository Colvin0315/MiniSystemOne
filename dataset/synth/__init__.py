"""合成数据包：`lexicon` 是词表事实来源，各生成器在此之上渲染 state。

**注册表集中在这里，新生成器只加一行。** `build_dataset.py` 与
`audit_synthetic.py` 都从 `GENERATORS` 取，避免出现"构建时用了一套、审计时用了
另一套"——那会让 audit 的结论指向一批并不存在于磁盘上的生成器，而且不报错。
"""
from dataset.synth.agent_trace_score import AgentTraceScore
from dataset.synth.banking_balance import BankingBalance
from dataset.synth.base import GEN_VERSION, assert_split_disjoint  # noqa: F401
from dataset.synth.calendar_slot import CalendarSlot
from dataset.synth.refund_policy import RefundPolicy
from dataset.synth.security_gate import SecurityGate
from dataset.synth.tool_router import ToolRouter

GENERATORS = (BankingBalance, ToolRouter, AgentTraceScore, SecurityGate,
              RefundPolicy, CalendarSlot)


def build_all(seed=0):
    """实例化全部生成器。seed 相同则 split 划分与样本都确定。"""
    return [cls(seed=seed) for cls in GENERATORS]
