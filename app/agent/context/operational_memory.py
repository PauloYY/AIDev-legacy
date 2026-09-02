from collections import deque
from dataclasses import dataclass
from typing import Any

from app.tools.registry import ToolRegistry


@dataclass
class ActionRecord:
    iteration: int
    tool: str
    arguments: dict[str, Any]
    success: bool
    result_summary: str
    dependency: bool = False


class OperationalMemory:
    """Memória operacional determinística do agente.

    Complementa (não substitui) o RESUMO DO PROJETO gerado por LLM em
    ProjectSummary/ProjectSummaryUpdater. Esse resumo é útil como
    narrativa de alto nível, mas é reescrito por um modelo a cada
    iteração e pode esquecer, resumir mal ou distorcer detalhes.

    Esta classe não faz nenhuma chamada de LLM: guarda em Python puro
    (1) o histórico real das últimas ações executadas nesta run
    (tool, argumentos, sucesso/falha, resultado) e (2) consulta a lista
    real de arquivos do projeto no disco (via a tool list_files) toda
    vez que render() é chamado.
    """

    MAX_ACTIONS = 20
    MAX_RESULT_CHARS = 300
    MAX_FILES_LISTED = 300

    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self._actions: deque[ActionRecord] = deque(maxlen=self.MAX_ACTIONS)

    def reset(self) -> None:
        self._actions.clear()

    def record(self, iteration, tool, arguments, result, success, dependency=False) -> None:
        self._actions.append(ActionRecord(
            iteration=iteration, tool=tool, arguments=arguments,
            success=success, result_summary=self._summarize(result),
            dependency=dependency,
        ))

    def render(self, project_name: str) -> str:
        return (
            "HISTÓRICO DE AÇÕES (memória determinística — Python puro, "
            "sempre precisa, não depende da LLM lembrar):\n"
            f"{self.render_history()}\n\n"
            "ARQUIVOS ATUAIS DO PROJETO (lista real do disco, consultada "
            "agora, não é um resumo):\n"
            f"{self.render_files(project_name)}"
        )

    def render_history(self) -> str:
        if not self._actions:
            return "Nenhuma ação executada ainda nesta run."
        return "\n".join(self._format_action(a) for a in self._actions)

    def render_files(self, project_name: str) -> str:
        try:
            files = self.tools.execute("list_files", {"project_name": project_name})
        except Exception as error:
            return f"Não foi possível listar os arquivos: {error}"
        if not files:
            return "(projeto vazio)"
        files = sorted(files)
        omitted = 0
        if len(files) > self.MAX_FILES_LISTED:
            omitted = len(files) - self.MAX_FILES_LISTED
            files = files[: self.MAX_FILES_LISTED]
        listing = "\n".join(f"- {p}" for p in files)
        if omitted:
            listing += f"\n... [+{omitted} arquivos omitidos]"
        return listing

    def _format_action(self, action: ActionRecord) -> str:
        status = "OK" if action.success else "FALHOU"
        kind = "dependency" if action.dependency else "task"
        return f"[{action.iteration}] ({kind}, {status}) {action.tool}({action.arguments}) -> {action.result_summary}"

    def _summarize(self, result: Any) -> str:
        text = str(result)
        if len(text) <= self.MAX_RESULT_CHARS:
            return text
        omitted = len(text) - self.MAX_RESULT_CHARS
        return f"{text[:self.MAX_RESULT_CHARS]}... [+{omitted} chars]"