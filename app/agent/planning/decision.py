from dataclasses import dataclass
from enum import Enum

from app.agent.execution.task import Task


class DecisionAction(Enum):
    TASK = "task"
    FINISH = "finish"
    FAIL = "fail"


@dataclass
class Decision:
    action: DecisionAction
    task: Task | None = None
    content: str | None = None
    reason: str | None = None