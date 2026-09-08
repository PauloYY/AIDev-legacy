from collections import deque
from dataclasses import dataclass
from hashlib import sha1
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
    # Etapa 2C: limite seguro para valores de argumentos exibidos no
    # histórico. Impede que uma ação individual insira dezenas de KB
    # no contexto do Planner (ex.: write_file com 10-50 KB). Só afeta
    # a REPRESENTAÇÃO do histórico — os argumentos reais do Executor
    # continuam intactos em ActionRecord.arguments.
    MAX_ARG_VALUE_CHARS = 200

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


    def render(
        self, project_name: str, files_text: str | None = None
    ) -> str:
        # files_text (Etapa 4): listagem pré-computada (ex.: compacta
        # "sem alterações"). None = comportamento legado integral.
        files_section = (
            files_text
            if files_text is not None
            else self.render_files(project_name)
        )
        return (
            "ACTION HISTORY (deterministic memory — pure Python, "
            "always accurate, does not depend on the LLM remembering):\n"
            f"{self.render_history()}\n\n"
            f"{self.render_known_commands()}\n\n"
            "CURRENT PROJECT FILES (real on-disk list, queried now, "
            "not a summary):\n"
            f"{files_section}"
        )

    def render_known_commands(self) -> str:
        test_line = (
            f"- Test: {self._known_test_command}"
            if self._known_test_command
            else "- Test: not discovered yet"
        )
        build_line = (
            f"- Build: {self._known_build_command}"
            if self._known_build_command
            else "- Build: not discovered yet"
        )

        return (
            "KNOWN WORKING COMMANDS (reuse these — "
            "only try a different command if one stops working "
            "after a code change):\n"
            f"{test_line}\n"
            f"{build_line}"
        )

    def render_history(self) -> str:
        if not self._actions:
            return "No actions executed yet in this run."
        return "\n".join(self._format_action(a) for a in self._actions)

    # ---------- Etapa 5: janela de histórico com preservação ----------

    # Política de janela (justificativa, não número mágico):
    # o Planner decide a PRÓXIMA ação a partir de (a) estado atual
    # (arquivos/summary/checklists, sempre integrais) + (b) causalidade
    # recente (o que foi feito e o que resultou) + (c) falhas que ainda
    # exigem correção. Cada task típica gera 1-3 registros (task +
    # 0-2 dependencies); 8 ações ≈ 3-4 ciclos completos decidir→
    # executar→observar, suficiente para detectar repetição imediata,
    # entender a última mudança e o último teste. Sucessos antigos já
    # estão materializados no estado atual (arquivos no disco, resumo,
    # checklist marcado) — reenviá-los é duplicação. Falhas antigas
    # fora da janela são preservadas explicitamente abaixo, porque um
    # erro não corrigido continua relevante mesmo depois de outras
    # ações.
    MAX_HISTORY_RENDERED_COMPACT = 8
    MAX_OLDER_FAILURES_KEPT = 2

    def render_history_compact(self) -> str:
        """Últimas N ações + falhas antigas preservadas (Etapa 5).

        Nunca remove: ações recentes (causalidade imediata), a falha
        mais recente fora da janela (pode explicar o estado atual) e
        o último teste/build com falha (o que precisa ser corrigido).
        Remove: sucessos antigos já refletidos no estado atual
        (arquivos/resumo/checklist). Retorna também linha de omissão
        explícita com a contagem — nunca omite silenciosamente.
        """
        if not self._actions:
            return "No actions executed yet in this run."

        actions = list(self._actions)
        window = actions[-self.MAX_HISTORY_RENDERED_COMPACT:]
        older = actions[:-self.MAX_HISTORY_RENDERED_COMPACT]

        lines = [self._format_action(a) for a in window]

        if older:
            # Falhas fora da janela que ainda importam: as mais
            # recentes primeiro, priorizando run_command de teste/build
            # (o que bloqueia finish) e depois qualquer falha.
            older_failures = [a for a in older if not a.success]
            kept: list = []
            test_failures = [
                a for a in older_failures
                if a.tool == "run_command"
            ]
            for candidate in test_failures + older_failures:
                if candidate not in kept:
                    kept.append(candidate)
                if len(kept) >= self.MAX_OLDER_FAILURES_KEPT:
                    break
            omitted = len(older) - len(kept)
            lines.append(
                f"... [{omitted} earlier action(s) omitted: "
                f"successes already reflected in the current state]"
            )
            for action in kept:
                lines.append(
                    f"(earlier failure preserved) "
                    f"{self._format_action(action)}"
                )

        return "\n".join(lines)

    def render_compact(
        self, project_name: str, files_text: str | None = None
    ) -> str:
        """Bloco de memória compacto (Etapa 5).

        Mesmas seções de render(), com histórico em janela
        (render_history_compact). Lista de arquivos, comandos
        conhecidos e formato geral inalterados — só o volume do
        histórico muda. `render()` legado segue intacto.
        """
        files_section = (
            files_text
            if files_text is not None
            else self.render_files(project_name)
        )
        return (
            "ACTION HISTORY (deterministic memory — pure Python, "
            "always accurate, does not depend on the LLM remembering):\n"
            f"{self.render_history_compact()}\n\n"
            f"{self.render_known_commands()}\n\n"
            "CURRENT PROJECT FILES (real on-disk list, queried now, "
            "not a summary):\n"
            f"{files_section}"
        )

    def render_files(self, project_name: str) -> str:
        try:
            files = self.tools.execute("list_files", {"project_name": project_name})
        except Exception as error:
            return f"Could not list files: {error}"
        if not files:
            return "(empty project)"
        return self._format_listing(sorted(files))

    def render_files_state(
        self, project_name: str, last_state: str | None = None
    ) -> tuple[str, str | None]:
        """Listagem estrutural com detecção de mudança (Etapa 4).

        Retorna (texto, estado). O estado é o hash da listagem
        COMPLETA ordenada (content-addressed): se nada mudou desde
        `last_state`, o texto é uma linha compacta explícita em vez
        da lista integral — o Planner continua sabendo quantos
        arquivos existem e que a estrutura está intacta. Mudanças de
        CONTEÚDO com o mesmo conjunto de paths não alteram o estado
        (a lista nunca teve conteúdos); essas chegam ao Planner via
        resultado da task, resumo e histórico, como antes. Em erro de
        listagem, retorna o texto de erro e estado None (força nova
        tentativa integral na próxima vez). `render()` segue inalterado.
        """

        try:
            files = self.tools.execute(
                "list_files", {"project_name": project_name})
        except Exception as error:
            return (
                f"Could not list files: {error}", None)
        file_list = sorted(files) if files else []
        state = sha1(repr(file_list).encode("utf-8")).hexdigest()
        if last_state is not None and state == last_state:
            return (
                f"(no changes since last check — "
                f"{len(file_list)} file(s))",
                state,
            )
        if not file_list:
            return "(empty project)", state
        return self._format_listing(file_list), state

    def _format_listing(self, files: list) -> str:
        files = list(files)
        omitted = 0
        if len(files) > self.MAX_FILES_LISTED:
            omitted = len(files) - self.MAX_FILES_LISTED
            files = files[: self.MAX_FILES_LISTED]
        listing = "\n".join(f"- {p}" for p in files)
        if omitted:
            listing += f"\n... [+{omitted} files omitted]"
        return listing

    def _format_action(self, action: ActionRecord) -> str:
        status = "OK" if action.success else "FAILED"
        kind = "dependency" if action.dependency else "task"
        args_repr = self._format_arguments_for_history(
            action.tool, action.arguments
        )
        return f"[{action.iteration}] ({kind}, {status}) {action.tool}({args_repr}) -> {action.result_summary}"

    def _format_arguments_for_history(
        self, tool: str, arguments: dict[str, Any] | None
    ) -> str:
        """Representação compacta dos argumentos para o histórico.

        Regra principal (Etapa 2C): o histórico registra O QUE aconteceu,
        não repete o conteúdo inteiro do que foi escrito.

        - write_file.content NUNCA aparece (vira "<omitted: N chars>").
        - edit_file.old_text/new_text seguem a mesma regra.
        - Qualquer valor string maior que MAX_ARG_VALUE_CHARS é truncado
          com indicador "[+N chars]".
        - Valores não-string com repr muito longo são resumidos do mesmo
          jeito.
        - Nunca muta o dict original (cria um novo dict compacto).
        """
        if arguments is None:
            return "{}"
        if not isinstance(arguments, dict):
            return self._truncate_value(str(arguments))

        compact: dict[str, Any] = {}
        for key, value in arguments.items():
            if (tool == "write_file" and key == "content") or (
                tool == "edit_file" and key in ("old_text", "new_text")
            ):
                if value is None:
                    compact[key] = "<empty>"
                else:
                    text = value if isinstance(value, str) else str(value)
                    if len(text) == 0:
                        compact[key] = "<empty>"
                    else:
                        compact[key] = f"<omitted: {len(text)} chars>"
                continue

            if isinstance(value, str):
                if len(value) > self.MAX_ARG_VALUE_CHARS:
                    omitted = len(value) - self.MAX_ARG_VALUE_CHARS
                    compact[key] = (
                        f"{value[:self.MAX_ARG_VALUE_CHARS]}"
                        f"... [+{omitted} chars]"
                    )
                else:
                    compact[key] = value
                continue

            # Não-string: preserva se o repr for pequeno, senão resume.
            try:
                rep = repr(value)
            except Exception:
                rep = str(value)
            if len(rep) > self.MAX_ARG_VALUE_CHARS + 20:
                text = str(value)
                omitted = len(text) - self.MAX_ARG_VALUE_CHARS
                compact[key] = (
                    f"{text[:self.MAX_ARG_VALUE_CHARS]}"
                    f"... [+{omitted} chars]"
                )
            else:
                compact[key] = value

        return repr(compact)

    def _truncate_value(self, text: str) -> str:
        if len(text) <= self.MAX_ARG_VALUE_CHARS:
            return text
        omitted = len(text) - self.MAX_ARG_VALUE_CHARS
        return f"{text[:self.MAX_ARG_VALUE_CHARS]}... [+{omitted} chars]"

    def _summarize(self, result: Any) -> str:
        text = str(result)
        if len(text) <= self.MAX_RESULT_CHARS:
            return text
        omitted = len(text) - self.MAX_RESULT_CHARS
        return f"{text[:self.MAX_RESULT_CHARS]}... [+{omitted} chars]"