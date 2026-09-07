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
        # Etapa 6: o Executor NÃO é bypassado (1 chamada LLM por
        # execução + retries, como antes). Só o prompt encolhe quando
        # AIDEV_COMPACT_EXECUTOR=1; a decisão do Planner (tool, args,
        # dependencies, ordem, arquivos) vai integral nos dois modos.
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
                "O executor tentou executar ferramentas diretamente. "
                "O TaskDecisionMaker deve retornar somente um objeto JSON em texto."
            )

        if response.content is None:
            raise LLMInvalidResponseError(
                "O executor não retornou conteúdo. "
                "A resposta deve conter somente um objeto JSON em texto."
            )

        return self.parser.parse(response.content)

    @staticmethod
    def _build_prompt(objective: str, task: Task, context: str) -> str:
        """Prompt integral do Executor (legado, byte a byte)."""
        return f"""
Você é o executor de uma tarefa de desenvolvimento.

Sua função é decidir como executar a task fornecida,
utilizando as informações disponíveis no contexto.

OBJETIVO:
{objective}

TASK:
Tool: {task.tool}

ARGUMENTOS:
{task.arguments}

CONTEXTO DAS DEPENDÊNCIAS:
{context}

REGRAS:
- Retorne SOMENTE JSON válido.
- NÃO execute ferramentas.
- NÃO produza tool calls.
- Use exclusivamente a tool indicada na task.
- Não crie novas tasks.
- Não crie dependencies.
- Não altere a finalidade da task.
- Os argumentos devem ser válidos para a tool.
- Para write_file, forneça o conteúdo completo do arquivo.
- Para read_file, mantenha os argumentos necessários para leitura.

FORMATO:
{{
    "tool": "{task.tool}",
    "arguments": {{
        ...
    }}
}}

Retorne SOMENTE o JSON.
"""

    @staticmethod
    def _build_prompt_compact(
        objective: str, task: Task, context: str
    ) -> str:
        """Prompt compacto do Executor (Etapa 6, semântico, sem corte).

        Mantém integralmente: objetivo, decisão do Planner (tool,
        argumentos completos, dependencies via contexto), restrições
        (só JSON, sem tool calls, só a tool indicada, sem novas
        tasks/dependencies, sem mudar a finalidade, args válidos,
        write_file com conteúdo completo) e formato. Remove apenas
        redundância: cabeçalho em 1 linha, regras fundidas (as 3
        proibições de execução/criação viram 2 bullets) e fecho
        duplicado ("Retorne SOMENTE o JSON" aparecia 2x).
        """
        return f"""Você é o executor da task abaixo. Decida os argumentos finais de execução usando o contexto.

OBJETIVO:
{objective}

TASK:
Tool: {task.tool}

ARGUMENTOS:
{task.arguments}

CONTEXTO DAS DEPENDÊNCIAS:
{context}

REGRAS:
- Retorne SOMENTE JSON válido. NÃO execute ferramentas nem produza tool calls.
- Use exclusivamente a tool indicada; não crie tasks/dependencies nem altere a finalidade.
- Argumentos válidos para a tool. Para write_file, forneça o conteúdo completo do arquivo.

FORMATO:
{{
    "tool": "{task.tool}",
    "arguments": {{
        ...
    }}
}}
"""