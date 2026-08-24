from dataclasses import dataclass


@dataclass
class ExecutionDecision:
    tool: str
    arguments: dict