from dataclasses import dataclass


@dataclass
class Dependency:
    tool: str
    arguments: dict