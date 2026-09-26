from app.agent.execution.execution_decision import ExecutionDecision
from app.agent.execution.execution_decision_parser import ExecutionDecisionParser
from app.agent.execution.task import Task
from app.exceptions import LLMInvalidResponseError
from app.llm.client import LLMClient
from app.llm.models import Message


class TaskDecisionMaker:

    def __init__(
        self,
        llm: LLMClient,
        parser: ExecutionDecisionParser,
    ):
        self.llm = llm
        self.parser = parser

    def decide(
        self,
        objective: str,
        task: Task,
        context: str,
        iteration: int | None = None,
    ) -> ExecutionDecision:
        from app.config import Config

        if bool(getattr(Config, "compact_executor", False)):
            try:
                prompt = self._build_prompt_compact(
                    objective, task, context)
            except Exception:
                prompt = self._build_prompt(objective, task, context)
        else:
            prompt = self._build_prompt(objective, task, context)

        response = self.llm.generate(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ],
            component="TaskDecisionMaker",
            iteration=iteration,
        )

        if response.tool_calls:
            raise LLMInvalidResponseError(
                "The executor tried to execute tools directly. "
                "The TaskDecisionMaker must return only a JSON object as text."
            )

        if response.content is None:
            raise LLMInvalidResponseError(
                "The executor returned no content. "
                "The response must contain only a JSON object as text."
            )

        return self.parser.parse(response.content)

    @staticmethod
    def _build_prompt(objective: str, task: Task, context: str) -> str:
        """Prompt integral do Executor (legado, byte a byte)."""
        return f"""
You are the executor of a software development task.

Your role is to decide how to execute the given task,
using the information available in the context.

OBJECTIVE:
{objective}

TASK:
Tool: {task.tool}

ARGUMENTS:
{task.arguments}

DEPENDENCY CONTEXT:
{context}

RULES:
- Return ONLY valid JSON.
- Do NOT execute tools.
- Do NOT produce tool calls.
- Use exclusively the tool indicated in the task.
- Do not create new tasks.
- Do not create dependencies.
- Do not change the purpose of the task.
- Arguments must be valid for the tool.
- For write_file, provide the full file content.
- For read_file, keep the arguments needed for reading.

FORMAT:
{{
    "tool": "{task.tool}",
    "arguments": {{
        ...
    }}
}}

Return ONLY the JSON.
"""

    @staticmethod
    def _build_prompt_compact(
        objective: str, task: Task, context: str
    ) -> str:
        """Prompt compacto do Executor (semântico, sem corte).

        Mantém integralmente: objetivo, decisão do Planner (tool,
        argumentos completos, dependencies via contexto), restrições
        (só JSON, sem tool calls, só a tool indicada, sem novas
        tasks/dependencies, sem mudar a finalidade, args válidos,
        write_file com conteúdo completo) e formato. Remove apenas
        redundância: cabeçalho em 1 linha, regras fundidas (as 3
        proibições de execução/criação viram 2 bullets) e fecho
        duplicado ("Retorne SOMENTE o JSON" aparecia 2x).
        """
        return f"""You are the executor of the task below. Decide the final execution arguments using the context.

OBJECTIVE:
{objective}

TASK:
Tool: {task.tool}

ARGUMENTS:
{task.arguments}

DEPENDENCY CONTEXT:
{context}

RULES:
- Return ONLY valid JSON. Do NOT execute tools or produce tool calls.
- Use exclusively the indicated tool; do not create tasks/dependencies or change the purpose.
- Valid arguments for the tool. For write_file, provide the full file content.

FORMAT:
{{
    "tool": "{task.tool}",
    "arguments": {{
        ...
    }}
}}
"""
