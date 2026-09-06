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

    # A cada quantas iterações uma investigação avulsa (read_file/
    # list_files/find_references como task principal, fora do passe
    # livre pós-falha de teste/build) é liberada. Isso convive com a
    # exceção de last_run_command_failed_test_or_build() — não a
    # substitui: mesmo dentro da janela "sem orçamento", uma falha de
    # teste/build ainda libera investigação normalmente.
    INVESTIGATION_BUDGET_PERIOD = 5

    # Palavras-chave usadas tanto para lembrar o comando de teste/build
    # que funcionou quanto para o gate de "libera read_file solo logo
    # após um teste falhar". "test"/"build" sozinhos não bastam —
    # runners populares como jest, mocha, pytest (via "test" mas
    # cobrindo por garantia) não contêm essas substrings.
    TEST_COMMAND_KEYWORDS = (
        "test", "jest", "mocha", "vitest", "pytest", "rspec",
        "phpunit", "karma", "jasmine", "spec", "tox",
    )
    BUILD_COMMAND_KEYWORDS = (
        "build", "compile", "webpack", "tsc", "make",
    )

    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self._actions: deque[ActionRecord] = deque(maxlen=self.MAX_ACTIONS)
        self._known_test_command: str | None = None
        self._known_build_command: str | None = None
        self._last_investigation_iteration: int | None = None

    def reset(self) -> None:
        self._actions.clear()
        self._known_test_command = None
        self._known_build_command = None
        self._last_investigation_iteration = None

    def record(self, iteration, tool, arguments, result, success, dependency=False) -> None:
        self._actions.append(ActionRecord(
            iteration=iteration, tool=tool, arguments=arguments,
            success=success, result_summary=self._summarize(result),
            dependency=dependency,
        ))

        if success and tool == "run_command":
            self._remember_known_command(arguments)

    def is_test_or_build_command(self, command: str) -> bool:
        """Retorna True se o texto do comando parece ser de teste ou
        build, pela heurística de palavras-chave compartilhada. Público
        porque o Runner também usa para decidir se deve gerar/limpar
        o checklist de erros ao redor de um run_command."""

        lowered = str(command).lower()

        return any(
            k in lowered
            for k in (*self.TEST_COMMAND_KEYWORDS, *self.BUILD_COMMAND_KEYWORDS)
        )

    def _remember_known_command(self, arguments: dict[str, Any]) -> None:
        """Guarda o último comando de teste/build bem-sucedido.

        O Planner ficava "redescobrindo" o comando certo a cada
        iteração (ex.: alternando entre 'npm test', 'node --test
        tests/*.test.js', 'node --test a.test.js b.test.js'), mesmo já
        tendo encontrado um que funcionava. Isso é Python puro — não
        depende da LLM lembrar corretamente.
        """

        command = str(arguments.get("command", "")).strip()

        if not command:
            return

        lowered = command.lower()

        if any(k in lowered for k in self.TEST_COMMAND_KEYWORDS):
            self._known_test_command = command
        elif any(k in lowered for k in self.BUILD_COMMAND_KEYWORDS):
            self._known_build_command = command

    def last_run_command_failed_test_or_build(self) -> bool:
        """Retorna True se a ação mais recente registrada foi um
        run_command que falhou (exit code != 0) e cujo comando parece
        ser de teste/build (mesma heurística usada para lembrar
        comandos que funcionam).

        Usado pelo Runner para liberar, só nesse caso específico, um
        read_file/list_files/find_references avulso como task
        principal — depois que um teste/build quebra, o Planner
        geralmente precisa investigar o arquivo/stack trace antes de
        conseguir corrigir, e forçar isso a virar dependency de uma
        write_file "às cegas" (sem ainda saber o que corrigir) não
        faz sentido.
        """

        if not self._actions:
            return False

        last = self._actions[-1]

        if last.tool != "run_command" or last.success:
            return False

        return self.is_test_or_build_command(
            str(last.arguments.get("command", ""))
        )

    def investigation_budget_available(self, iteration: int) -> bool:
        """Orçamento periódico de investigação avulsa, independente de
        ter havido falha de teste/build.

        Libera 1 investigação a cada INVESTIGATION_BUDGET_PERIOD
        iterações — cobre casos legítimos de "preciso olhar algo antes
        de decidir" que não vêm logo após um teste/build quebrado (ex.:
        início de uma tarefa complexa, ou revisão de um arquivo antes
        de uma refatoração maior). Não consome sozinho: só é marcado
        como usado quando o Runner efetivamente aprova uma investigação
        por este motivo (veja consume_investigation_budget).
        """

        if self._last_investigation_iteration is None:
            return True

        return (
            iteration - self._last_investigation_iteration
            >= self.INVESTIGATION_BUDGET_PERIOD
        )

    def consume_investigation_budget(self, iteration: int) -> None:
        self._last_investigation_iteration = iteration


    def render(self, project_name: str) -> str:
        return (
            "HISTÓRICO DE AÇÕES (memória determinística — Python puro, "
            "sempre precisa, não depende da LLM lembrar):\n"
            f"{self.render_history()}\n\n"
            f"{self.render_known_commands()}\n\n"
            "ARQUIVOS ATUAIS DO PROJETO (lista real do disco, consultada "
            "agora, não é um resumo):\n"
            f"{self.render_files(project_name)}"
        )

    def render_known_commands(self) -> str:
        test_line = (
            f"- Teste: {self._known_test_command}"
            if self._known_test_command
            else "- Teste: ainda não descoberto"
        )
        build_line = (
            f"- Build: {self._known_build_command}"
            if self._known_build_command
            else "- Build: ainda não descoberto"
        )

        return (
            "COMANDOS CONHECIDOS QUE JÁ FUNCIONARAM (reutilize estes — "
            "só tente um comando diferente se este parar de funcionar "
            "depois de uma mudança de código):\n"
            f"{test_line}\n"
            f"{build_line}"
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