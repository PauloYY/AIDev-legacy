import json

from app.llm.client import LLMClient
from app.llm.models import Message


class ProjectAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(
        self,
        project_name: str,
        files: list[str],
    ) -> str:
        prompt = self._build_prompt(
            project_name,
            files,
        )

        response = self.llm.generate(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ],
            component="ProjectAnalyzer",
        )

        return response.content

    def _build_prompt(
        self,
        project_name: str,
        files: list[str],
    ) -> str:
        files_json = json.dumps(
            files,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are the project analyzer of a development agent.

Analyze the project structure below and produce an initial summary
to be used later by another agent to plan tasks.

PROJECT:
{project_name}

FILES:
{files_json}

RULES:
- Return only the summary content.
- Do not use JSON.
- Do not invent information about the files.
- Base your answer only on the given structure.
- State that the file contents have not been analyzed yet.
- Be concise.
- The result will be saved directly to .aidev/summary.md.

The summary must contain:

# Project Summary

## Structure

List the relevant files.

## Understanding

Describe only what can be inferred from the structure.

## Notes

State that the files still need to be analyzed when necessary.
"""