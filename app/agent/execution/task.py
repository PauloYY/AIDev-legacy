from dataclasses import dataclass, field

from app.agent.planning.dependency import Dependency


@dataclass
class Task:
    tool: str
    arguments: dict
    dependencies: list[Dependency] = field(default_factory=list)