import json
import logging

from app.agent.planning.decision import Decision
from app.agent.planning.decision_parser import DecisionParser
from app.llm.client import LLMClient
from app.tools.registry import ToolRegistry
from app.llm.models import Message


logger = logging.getLogger(__name__)


class Planner:

    def __init__(
        self,
        llm: LLMClient,
        parser: DecisionParser,
        tools: ToolRegistry,
    ):
        self.llm = llm
        self.parser = parser
        self.tools = tools
        self._cached_tools_context: str | None = None
        self._cached_tools_names: tuple[str, ...] = ()
        # Último conteúdo bruto retornado pela LLM (mesmo quando o
        # parse falha). O Runner usa para montar o short repair prompt
        # sem precisar reenviar o contexto completo. Só diagnóstico.
        self.last_raw_response: str | None = None

    def plan(
        self,
        objective: str,
        context: str = "",
        iteration: int | None = None,
        request_type: str = "normal",
    ) -> Decision:
        """Decisão normal do Planner.

        Legado: prompt completo via _build_prompt().
        Etapa 5 (AIDEV_COMPACT_PLANNER=1): prompt compacto via
        _build_prompt_compact() — mesmo objetivo, mesmas regras
        essenciais e mesmas tools, com representação mais curta.
        Short repair nunca passa por aqui (usa plan_with_prompt).
        """

        from app.config import Config

        use_compact = bool(getattr(Config, "compact_planner", False))
        if use_compact:
            try:
                prompt = self._build_prompt_compact(
                    objective,
                    context,
                )
            except Exception as error:
                logger.warning(
                    "Falha no prompt compacto do Planner "
                    "(usando prompt completo): %s",
                    error,
                )
                use_compact = False
                prompt = self._build_prompt(
                    objective,
                    context,
                )
        else:
            prompt = self._build_prompt(
                objective,
                context,
            )

        # Etapa 2B: instrumentação observacional — mede cada componente
        # do prompt sem alterar conteúdo, ordem ou decisão.
        context_breakdown = None
        try:
            if use_compact:
                sections = self.build_compact_prompt_sections(
                    objective, context)
            else:
                sections = self.build_prompt_sections(objective, context)
            context_breakdown = self.measure_prompt_sections(sections)
        except Exception as error:
            # A medição nunca pode quebrar o Planner.
            logger.warning(
                "Falha na instrumentação do contexto do Planner: %s",
                error,
            )
            context_breakdown = None

        return self.plan_with_prompt(
            prompt=prompt,
            iteration=iteration,
            request_type=request_type,
            context_breakdown=context_breakdown,
        )

    def plan_with_prompt(
        self,
        prompt: str,
        iteration: int | None = None,
        request_type: str = "normal",
        context_breakdown: dict[str, dict[str, int]] | None = None,
    ) -> Decision:
        """Executa UMA chamada de decisão com um prompt já montado.

        Caminho compartilhado entre a decisão normal (prompt completo)
        e o short repair (prompt curto). Não decide nada sozinho: só
        chama a LLM, valida o envelope e delega o parse ao
        DecisionParser — as mesmas regras de antes.
        """

        try:
            response = self.llm.generate(
                messages=[
                    Message(
                        role="user",
                        content=prompt,
                    )
                ],
                component="Planner",
                iteration=iteration,
                context_breakdown=context_breakdown,
                request_type=request_type,
            )
        except TypeError:
            try:
                # Compatibilidade com doubles antigos sem `request_type`.
                response = self.llm.generate(
                    messages=[
                        Message(
                            role="user",
                            content=prompt,
                        )
                    ],
                    component="Planner",
                    iteration=iteration,
                    context_breakdown=context_breakdown,
                )
            except TypeError:
                # Compatibilidade com doubles de LLM antigos que não
                # aceitam o kwarg de instrumentação (ex.: FakeLLM em
                # testes legados).
                response = self.llm.generate(
                    messages=[
                        Message(
                            role="user",
                            content=prompt,
                        )
                    ],
                    component="Planner",
                    iteration=iteration,
                )

        # Guarda o bruto ANTES de validar: se o parse falhar, o Runner
        # ainda consegue mostrar a decisão anterior no repair prompt.
        try:
            self.last_raw_response = response.content
        except Exception:
            self.last_raw_response = None

        if response.tool_calls:
            raise ValueError(
                "O Planner tentou executar ferramentas diretamente. "
                "O Planner deve retornar somente um objeto JSON em texto."
            )

        if response.content is None:
            raise ValueError(
                "O Planner não retornou conteúdo. "
                "A resposta deve conter somente um objeto JSON em texto."
            )

        return self.parser.parse(response.content)

    # ---------- Fase 3 Etapa 2: short repair prompt ----------

    # Marcadores (PT/EN) usados para escolher a dica direcionada. São
    # os mesmos textos que o validador/parser já emitem — nenhuma regra
    # nova, só classificação para montar o reparo mínimo.
    _HINT_DEPENDENCY_ONLY = "só pode ser usada como dependency"
    _HINT_UNKNOWN_TOOL = "Tool não encontrada"
    _HINT_BAD_DEPENDENCY = "não pode ser usada como dependency"
    _HINT_SCHEMA = (
        "required", "propriedade", "properties", "schema", "Schema",
        "argumento", "additional",
    )
    # Versão minúscula (inclui variações com maiúscula inicial, como
    # "Argumento desconhecido ... Argumentos válidos", do schema).
    _HINT_SCHEMA_LOWER = (
        "required", "propriedade", "properties", "schema", "argumento",
        "argumentos", "obrigatóri", "válido", "desconhecido",
        "additional", "tipo",
    )
    _HINT_JSON = "JSON inválido"
    _HINT_TOOL_CALLS = "executar ferramentas diretamente"
    _HINT_NO_CONTENT = "não retornou conteúdo"
    _HINT_BAD_ACTION = "Ação de decisão inválida"

    MAX_REPAIR_PREVIOUS_CHARS = 1500
    MAX_REPAIR_ERROR_CHARS = 500

    def full_prompt_chars(self, objective: str, context: str) -> int:
        """Tamanho aproximado do prompt completo (só mede, p/ o trace).

        Etapa 5: mede o que será realmente enviado — prompt compacto
        quando AIDEV_COMPACT_PLANNER=1, legado caso contrário.
        """
        try:
            from app.config import Config

            if bool(getattr(Config, "compact_planner", False)):
                try:
                    return len(self._build_prompt_compact(
                        objective or "", context or ""))
                except Exception:
                    pass
            return len(self._build_prompt(objective or "", context or ""))
        except Exception:
            return 0

    def build_repair_prompt(
        self,
        error: str,
        raw_response: str | None = None,
        tool_name: str | None = None,
    ) -> str:
        """Monta o short repair prompt para UM retry interno do Planner.

        Contém somente: a decisão anterior (truncada), o erro de
        validação e a dica direcionada ao erro. NÃO inclui resumo do
        projeto, histórico, lista de arquivos, checklists, memória de
        erros nem o prompt estático — por isso custa uma fração do
        retry com contexto completo.
        """

        previous = (raw_response or "").strip()
        if len(previous) > self.MAX_REPAIR_PREVIOUS_CHARS:
            omitted = len(previous) - self.MAX_REPAIR_PREVIOUS_CHARS
            previous = (
                f"{previous[:self.MAX_REPAIR_PREVIOUS_CHARS]}\n"
                f"...[truncated, {omitted} chars omitted]"
            )
        if not previous:
            previous = "(empty response)"

        error_text = str(error or "").strip()
        if len(error_text) > self.MAX_REPAIR_ERROR_CHARS:
            omitted = len(error_text) - self.MAX_REPAIR_ERROR_CHARS
            error_text = (
                f"{error_text[:self.MAX_REPAIR_ERROR_CHARS]}\n"
                f"...[truncated, {omitted} chars omitted]"
            )

        hint = self._repair_hint(error_text, tool_name)

        return (
            "You are the PLANNER of an autonomous software development "
            "agent. Your previous decision was INVALID — fix ONLY what "
            "the validation error points out.\n"
            "\n"
            "PREVIOUS DECISION:\n"
            f"{previous}\n"
            "\n"
            "VALIDATION ERROR:\n"
            f"{error_text}\n"
            "\n"
            "TARGETED FIX:\n"
            f"{hint}\n"
            "\n"
            "RULES FOR THIS REPAIR:\n"
            "- Return ONLY the corrected decision JSON "
            "(same format as a normal Planner decision).\n"
            "- Do NOT execute tools. Do NOT explain.\n"
            "- Keep the same task/tool/arguments unless the validation "
            "error requires changing them."
        )

    def _repair_hint(self, error_text: str, tool_name: str | None) -> str:
        """Escolhe a dica mínima para o erro (sem reenviar contexto)."""

        lowered = error_text.lower()

        if self._HINT_DEPENDENCY_ONLY in lowered:
            tool = tool_name or "read_file"
            return (
                f"The tool '{tool}' cannot be a standalone task. Fix it "
                "in ONE of these two ways:\n"
                "1. Attach it as a \"dependencies\" entry of the REAL "
                "action that needs the information (e.g. a write_file "
                "or run_command task), instead of a task alone; OR\n"
                "2. Only if this is a pure analysis objective or you "
                "are investigating a failed test/build, keep it as the "
                "main task but add \"investigation\": true to the task, "
                "e.g. {\"action\": \"task\", \"task\": {\"tool\": "
                f"\"{tool}\", \"arguments\": {{...}}, "
                "\"investigation\": true, \"dependencies\": []}}."
            )

        if self._HINT_UNKNOWN_TOOL.lower() in lowered:
            names = sorted(self.tools._tools.keys())
            return (
                "Unknown tool name. Use EXACTLY one of these available "
                f"tools: {', '.join(names)}. Fix ONLY the tool name "
                "(keep the rest unless it also violates a rule)."
            )

        if self._HINT_BAD_DEPENDENCY in lowered:
            return (
                "Only analysis tools may appear inside \"dependencies\". "
                "Remove the non-analysis entry from \"dependencies\" or "
                "move that action to be the main task tool."
            )

        if tool_name and any(
            marker in lowered for marker in self._HINT_SCHEMA_LOWER
        ):
            schema = self._compact_tool_schema(tool_name)
            if schema is not None:
                return (
                    "The arguments do not match the tool schema. Exact "
                    f"schema for '{tool_name}':\n"
                    f"{schema}\n"
                    "Fix ONLY the arguments to match \"required\" and "
                    "the property names."
                )

        if self._HINT_JSON.lower() in lowered:
            return (
                "The response was not valid JSON. Return a single JSON "
                "object with \"action\" equal to \"task\", \"finish\" "
                "or \"fail\" (see the normal decision formats)."
            )

        if self._HINT_TOOL_CALLS in lowered:
            return (
                "Never use tool calls. Return the decision as plain "
                "JSON text only."
            )

        if self._HINT_NO_CONTENT in lowered:
            return (
                "The response had no content. Return the decision as "
                "plain JSON text."
            )

        if self._HINT_BAD_ACTION.lower() in lowered:
            return (
                "Invalid action. Use \"action\": \"task\" (with a "
                "\"task\" object), \"finish\" (with \"content\") or "
                "\"fail\" (with \"reason\")."
            )

        return (
            "Return a valid decision JSON: {\"action\": \"task\", "
            "\"task\": {\"tool\": \"<available tool>\", "
            "\"arguments\": {...}, \"dependencies\": []}} — or "
            "\"finish\"/\"fail\". Fix exactly what the validation "
            "error describes."
        )

    def _compact_tool_schema(self, tool_name: str) -> str | None:
        """Schema mínimo de UMA tool (nome, required, propriedades)."""

        try:
            definition = self.tools.get(tool_name).definition
        except Exception:
            return None

        try:
            function = definition.get("function", {})
            params = function.get("parameters", {})
            properties = params.get("properties", {})

            compact_props: dict[str, str] = {}
            for key, spec in properties.items():
                if isinstance(spec, dict):
                    text = spec.get("description") or spec.get("type", "")
                else:
                    text = str(spec)
                compact_props[key] = str(text)[:160]

            return json.dumps(
                {
                    "name": function.get("name", tool_name),
                    "description": str(
                        function.get("description", ""))[:300],
                    "required": params.get("required", []),
                    "properties": compact_props,
                },
                ensure_ascii=False,
            )
        except Exception:
            return None

    def build_prompt_sections(
        self,
        objective: str | None,
        context: str | None,
    ) -> dict[str, str]:
        """Decompõe o prompt final em componentes nomeados (só observa).

        Retorna dict com chaves de PLANNER_COMPONENTS:
        static_template, objective, project_summary, objective_checklist,
        error_checklist, planner_error_memory, action_history,
        known_commands, file_list, task_context, execution_result,
        error_blocks, other_context.

        - static_template: prompt construído com objective/context vazios
          (inclui instruções fixas + definições de tools).
        - objective: texto do objetivo ("" se None).
        - demais: fatiamento do `context` via prompt_sections.
        Não altera o prompt final: a concatenação dos componentes equivale
        ao retorno de _build_prompt() a menos de separadores "\n\n".
        """
        from app.agent.planning.prompt_sections import split_planner_context

        if objective is None:
            objective_text = ""
        elif isinstance(objective, str):
            objective_text = objective
        else:
            objective_text = str(objective)

        if context is None:
            context_text = ""
        elif isinstance(context, str):
            context_text = context
        else:
            context_text = str(context)

        static_template_text = self._build_prompt("", "")

        dynamic = split_planner_context(context_text)

        sections: dict[str, str] = {
            "static_template": static_template_text,
            "objective": objective_text,
        }
        sections.update(dynamic)
        return sections

    @staticmethod
    def measure_prompt_sections(
        sections: dict[str, str | None],
    ) -> dict[str, dict[str, int]]:
        """Mede chars + tokens estimados por componente via estimate_tokens()."""
        from app.agent.planning.prompt_sections import measure_sections

        return measure_sections(sections)

    def _build_prompt(
        self,
        objective: str,
        context: str,
    ) -> str:

        return f"""You are the PLANNER of an autonomous software development agent.

Your role is to choose the NEXT action needed to achieve the objective.

RULES:

- Return ONLY valid JSON.
- Use only available tools. A task is a single action; dependencies gather info before it and use only analysis tools.
- "read_file", "list_files", "find_references" cannot be a standalone task — they are investigation tools, only allowed inside "dependencies". If you need a file's content before editing it, attach read_file as a dependency of the write_file/edit_file/run_command task; do not create a task just to read.
- EXCEPTION: after a failed "run_command" test/build, you MAY use read_file/list_files/find_references alone as the main task to investigate what broke (e.g. read the stack trace file) before knowing the fix. Outside this case, the rule above applies — do not chain multiple standalone investigations.
- ANOTHER EXCEPTION: if your OBJECTIVE is to analyze/investigate an existing project (without creating or modifying code), mark the task with "investigation": true. This explicitly allows read_file/list_files/find_references as the main action. Example:
  {{"action": "task", "task": {{"tool": "read_file", "arguments": {{...}}, "investigation": true}}}}
  Without this field, the task will be rejected by the validator.
- A task can have multiple dependencies — if you need to read multiple files before acting, attach them all at once instead of making separate tasks.
- For large content arguments (e.g. "content" of write_file), do NOT write the final content here — briefly describe what should be done (e.g. "implement the Client class with fields id, name, and status"). The EXECUTOR generates the complete file content, so repeating it here wastes tokens and increases the risk of truncated responses.
- For a small, localized change in an existing file, prefer "edit_file" (precise old_text → new_text, which must match exactly once) over rewriting the whole file with "write_file". Use "delete_file" only to remove a single file that is clearly unnecessary; never to remove directories or protected paths (.git, .aidev).
- TOOL SELECTION POLICY (a preference, not a hard rule — prefer the most precise tool appropriate for the operation): new file → write_file; existing file + localized change → edit_file; existing file + substantial rewrite → write_file; clearly unnecessary file → delete_file.

PROGRESS:

- Analyze context before deciding. Consider everything already executed.
- Do not repeat a task already completed successfully. Do not repeat the same tool with the same arguments without concrete justification.
- If an action failed, use the error to decide the next step. Each new task must produce real progress.
- Tasks marked "investigation": true are pure read/analysis actions — they do not mutate the project. They do not count as stagnation internally, but the Planner must not investigate indefinitely: after gathering sufficient information, make a decision (finish, action task, or fail).
- Before writing an import/require that references another project file, use the list_symbols tool on that file to confirm the exact exported name instead of guessing.
- If the objective is already complete, use "finish". Use "fail" only when continuation is impossible.

OBJECTIVE CHECKLIST:

- The context includes a CHECKLIST OF OBJECTIVES with numbered items, defined once at the start and never rewritten.
- Use it as a guide: follow the order of pending items instead of rediscovering what remains each iteration.
- When the task you are deciding now completes one or more checklist items, include the optional field "checklist_progress": [id, id, ...] in your decision (works in "task", "finish", or "fail" decisions) with the completed ids.
- Only mark an item when it has been truly accomplished (e.g. the write_file/run_command that implements it has been executed successfully), not when you are merely planning to do it.
- The checklist is a guide, not a straitjacket: if you discover an item is no longer needed or a new item is missing, follow the real state of the project — do not cling to the checklist at the cost of ignoring a real error.

ERROR CHECKLIST:

- When the context includes a CURRENT ERROR CHECKLIST, this means the last failed test/build run_command had each item as a concrete failure extracted automatically from the real output.
- Use it to know exactly what needs fixing, instead of re-reading the raw test output each time.
- This checklist is automatic: there is no field to mark an item as resolved. It disappears on its own once you run the test/build again and it passes; if it still fails, it is regenerated from scratch with the current failures. Do not invent that an item was fixed without first confirming by running the test again.
- When it appears, prioritize fixing these errors before moving to new OBJECTIVE CHECKLIST items.
- While any item exists on this checklist, "finish" is blocked — do not attempt to finish with pending test/build failures; fix and confirm the test/build passes first.

PROJECT COHERENCE:

- Preserve existing names and interfaces. Do not invent classes, functions, methods, attributes, or parameters that should exist in the project.
- If you need to use something defined in another file, confirm its definition using dependencies when needed. If you need to change an existing interface, consider its consumers before changing.
- The current project state takes priority over old summary information.

VALIDATION BEFORE FINISH:

- Writing code DOES NOT mean it works. Do not assume it is correct just because you wrote it.
- Before using "finish", use "run_command" to execute the program, tests, or build (whatever makes sense for the project language/framework) and confirm it actually meets the objective.
- Choose the command according to the project: "python main.py", "python -m pytest -q", "npm test", "npm run build", "node app.js", "gcc main.c -o main && ./main", "g++ main.cpp -o main && ./main", "javac Main.java && java Main", "go run .", "go test ./...", "cargo run", "cargo test", "ruby main.rb", "php main.php", "mvn test", "gradle test", etc. Use only commands whose runtime is available in the sandbox image.
- For Gradle projects, use "gradle" directly — NEVER "./gradlew" (the wrapper tries to download Gradle over the internet, and the sandbox has no network access).
- For Node.js/JavaScript projects: ALWAYS create "package.json" BEFORE running "npm install" or "npm test". Never run "npm install" on a project that does not yet have "package.json" — it fails and may leave an inconsistent "package-lock.json" behind, which hinders future installations even after the correct "package.json" exists.
- Avoid commands that do not terminate on their own, such as servers (e.g. "npm start", "flask run", "python -m http.server") — they will timeout and do not serve as validation.
- If execution shows an error, exception, incorrect output, or unexpected behavior, this is NOT a reason for "finish" — create a task to fix the problem.
- AFTER A TEST/EXECUTION FAILS, follow this reasoning before deciding the next action:
  1. Read the error message carefully: which file, function, or line does it point to? What is the likely cause (e.g. wrong name, incorrect type, inverted logic, missing import)?
  2. If the message already clearly indicates what is wrong, fix it DIRECTLY with "write_file" (or "edit_file" for a small localized fix — use "read_file" as a dependency of the fix task if you need to confirm the current content before editing — not as a separate task).
  3. Do NOT run the same test/command again without having changed any code — running again without changing anything always gives the same result and is not progress.
  4. After fixing, run the test again to confirm the problem is resolved.
- For interactive programs (with input()), use the "stdin" argument of "run_command" to simulate user inputs and validate the main flows.
- Only use "finish" after validating by real execution that the objective was achieved.

TOOLS AVAILABLE:

{self._build_tools_context()}

DECISION FORMATS:

TASK (normal action):
{{
    "action": "task",
    "task": {{
        "tool": "write_file",
        "arguments": {{
            "project_name": "test-project",
            "file_path": "result.py",
            "content": "brief description of what the file should contain (not the full content)"
        }},
        "dependencies": []
    }}
}}

TASK (investigation of existing project — add "investigation": true when the OBJECTIVE is to analyze/investigate without modifying code):
{{
    "action": "task",
    "task": {{
        "tool": "list_files",
        "arguments": {{
            "project_name": "my-project"
        }},
        "investigation": true,
        "dependencies": []
    }}
}}

TASK (with dependency — use when you need to see something before acting):
{{
    "action": "task",
    "task": {{
        "tool": "write_file",
        "arguments": {{
            "project_name": "test-project",
            "file_path": "src/service.js",
            "content": "brief description of the change, considering the current file content"
        }},
        "dependencies": [
            {{
                "tool": "read_file",
                "arguments": {{
                    "project_name": "test-project",
                    "file_path": "src/service.js"
                }}
            }}
        ]
    }}
}}

FINISH:
{{
    "action": "finish",
    "content": "Description of the final result."
}}

FAIL:
{{
    "action": "fail",
    "reason": "Reason for failure."
}}

"checklist_progress" is always optional — omit it when the current task does not complete any checklist item.

OBJECTIVE:

{objective}

CURRENT CONTEXT:

{context}
"""

    def _build_tools_context(self) -> str:
        names = tuple(sorted(self.tools._tools.keys()))

        if (
            self._cached_tools_context is not None
            and self._cached_tools_names == names
        ):
            return self._cached_tools_context

        result = json.dumps(
            self.tools.definitions,
            indent=2,
            ensure_ascii=False,
        )

        self._cached_tools_context = result
        self._cached_tools_names = names
        return result

    # ---------- Etapa 5: compactação inteligente do Planner ----------

    # Tetos do schema compacto (mais agressivos que _compact_tool_schema,
    # que usa 300/160 para hints de repair). O validador continua usando
    # os schemas INTEGRAIS — o prompt só precisa do suficiente para
    # gerar uma decisão válida: nome, required, nomes/tipos dos args e
    # a distinção entre tools. Regras de uso (quando validar, o que
    # evitar) vivem no template estático, não na descrição da tool
    # (deduplicação: run_command/check_project/list_symbols repetiam o
    # mesmo conselho nos dois lugares).
    COMPACT_TOOL_DESC_CHARS = 150
    COMPACT_TOOL_PROP_CHARS = 80

    def _build_tools_context_compact(self) -> str:
        """Schemas mínimos das tools (nome, required, props, desc curta).

        Reutiliza a mesma ideia de _compact_tool_schema() (Etapa 2),
        com tetos menores. Nunca remove nome/required/tipos/diferenças
        entre tools. O cache é separado do contexto integral.
        """
        names = tuple(sorted(self.tools._tools.keys()))
        cached = getattr(self, "_cached_compact_tools_context", None)
        cached_names = getattr(self, "_cached_compact_tools_names", ())
        if cached is not None and cached_names == names:
            return cached

        compact_defs: list[dict] = []
        for tool_name in names:
            try:
                definition = self.tools.get(tool_name).definition
                function = definition.get("function", {})
                params = function.get("parameters", {})
                properties = params.get("properties", {}) or {}
                compact_props: dict[str, str] = {}
                for key, spec in properties.items():
                    if isinstance(spec, dict):
                        prop_type = str(spec.get("type", ""))
                        prop_desc = str(spec.get("description") or "")
                        # "type + essencial da descrição" cabe em 80 chars
                        # e preserva o tipo (obrigatório p/ decisão válida).
                        text = (
                            f"{prop_type}: {prop_desc}"
                            if prop_desc else prop_type
                        )
                    else:
                        text = str(spec)
                    compact_props[key] = text[:self.COMPACT_TOOL_PROP_CHARS]
                compact_defs.append({
                    "name": function.get("name", tool_name),
                    "description": str(
                        function.get("description", "")
                    )[:self.COMPACT_TOOL_DESC_CHARS],
                    "required": params.get("required", []),
                    "properties": compact_props,
                })
            except Exception:
                # Tool ilegível nunca pode quebrar o Planner: cai para
                # o schema integral dessa tool.
                try:
                    compact_defs.append(
                        self.tools.get(tool_name).definition)
                except Exception:
                    continue

        # Sem indentação: a LLM lê JSON corrido sem perda; economiza
        # ~20% de whitespace repetido a cada chamada do Planner.
        result = json.dumps(
            compact_defs, ensure_ascii=False, separators=(",", ":"))
        self._cached_compact_tools_context = result
        self._cached_compact_tools_names = names
        return result

    def build_compact_prompt_sections(
        self,
        objective: str | None,
        context: str | None,
    ) -> dict[str, str]:
        """Decompõe o prompt COMPACTO como build_prompt_sections faz.

        Mesmas chaves; static_template aqui é _build_prompt_compact().
        Só observação — nunca altera o prompt enviado.
        """
        from app.agent.planning.prompt_sections import split_planner_context

        if objective is None:
            objective_text = ""
        elif isinstance(objective, str):
            objective_text = objective
        else:
            objective_text = str(objective)

        if context is None:
            context_text = ""
        elif isinstance(context, str):
            context_text = context
        else:
            context_text = str(context)

        static_template_text = self._build_prompt_compact("", "")

        dynamic = split_planner_context(context_text)

        sections: dict[str, str] = {
            "static_template": static_template_text,
            "objective": objective_text,
        }
        sections.update(dynamic)
        return sections

    def _build_prompt_compact(
        self,
        objective: str,
        context: str,
    ) -> str:
        """Template estático compacto (Etapa 5).

        Compactação SEMÂNTICA, não truncamento: cada seção do prompt
        integral tem um correspondente aqui com as mesmas regras
        normativas, em menos palavras. Preserva obrigatoriamente:
        formato de resposta (só JSON), tools disponíveis, limites das
        tools (dependency-only + exceções + investigation flag),
        checklist_progress, error checklist bloqueando finish,
        coerência (não inventar nomes, confirmar via dependencies,
        estado atual > resumo), validação antes do finish
        (run_command real, comandos por linguagem, Gradle sem wrapper,
        package.json antes de npm install/test, sem servidores,
        corrigir após falha sem repetir comando, stdin) e condições
        de finalização (finish só após validação real, fail só se
        impossível). Exemplos redundantes e repetições entre
        descrição de tool e template foram removidos (a distinção
        entre tools vive nos schemas compactos acima).
        """

        return f"""You are the PLANNER of an autonomous software development agent. Choose the NEXT action to achieve the objective.

RULES:
- Return ONLY valid JSON.
- One task = one action; dependencies gather info first and use only analysis tools.
- "read_file", "list_files", "find_references" are investigation tools: only inside "dependencies", never as standalone task. To edit a file, attach read_file as a dependency of the write_file/edit_file/run_command task.
- EXCEPTION: after a failed "run_command" test/build you MAY use them alone to investigate the breakage. ANOTHER EXCEPTION: objective is pure analysis (no code changes) — then add "investigation": true to the task, e.g. {{"action": "task", "task": {{"tool": "read_file", "arguments": {{...}}, "investigation": true}}}} (rejected without it).
- Attach all needed reads at once as multiple dependencies. For large args (write_file "content") describe briefly what to do — the EXECUTOR writes the full content.
- Small localized change in an existing file? Prefer "edit_file" (exact old_text→new_text, single match) over a full "write_file". Remove a clearly unnecessary file only with "delete_file" (files only; never directories or .git/.aidev).
- TOOL SELECTION POLICY (preference — the most precise tool wins): new file → write_file; existing + localized → edit_file; existing + substantial rewrite → write_file; clearly unnecessary file → delete_file.

PROGRESS:
- Consider everything already executed. Never repeat a completed task or the same tool+args without justification from new evidence.
- On failure, use the error for the next step; each task must bring real progress. "investigation": true tasks are read-only (no stagnation count) but do not investigate forever — act, finish or fail.
- Before importing from another file, confirm the exact exported name with list_symbols. If the objective is done, "finish"; "fail" only when impossible.

OBJECTIVE CHECKLIST:
- Context has a fixed numbered CHECKLIST; follow pending items in order. When the decided task completes items, add "checklist_progress": [ids] ("task"/"finish"/"fail"). Only mark truly done work, not plans. Checklist guides — real project state wins over it.

ERROR CHECKLIST:
- CURRENT ERROR CHECKLIST lists concrete failures from the last failed test/build run_command. Fix these before new objective items. It clears itself when tests pass; never claim fixed without re-running. While any item exists, "finish" is blocked.

PROJECT COHERENCE:
- Keep existing names/interfaces; never invent classes/functions/params. Confirm definitions via dependencies when needed. Current project state beats old summary text.

VALIDATION BEFORE FINISH:
- Writing code is not proof. Before "finish", run the program/tests/build with run_command (python main.py, python -m pytest -q, npm test/build, node app.js, gcc/go/cargo/ruby/php/mvn/gradle equivalents; Gradle: use "gradle", NEVER "./gradlew" — no network).
- Node: create "package.json" BEFORE "npm install"/"npm test". Never run servers that do not terminate (npm start, flask run) — use "stdin" of run_command for interactive programs.
- After a test/execution failure: read which file/line likely caused it; fix DIRECTLY with write_file (or edit_file for a small localized fix; read_file as dependency if needed); NEVER re-run the same command unchanged; re-run after fixing to confirm. Only "finish" after real execution proves the objective.

TOOLS AVAILABLE:

{self._build_tools_context_compact()}

DECISION FORMATS:

TASK: {{"action": "task", "task": {{"tool": "write_file", "arguments": {{"project_name": "test-project", "file_path": "result.py", "content": "brief description (not full content)"}}, "dependencies": []}}}}
INVESTIGATION (analysis objective only): {{"action": "task", "task": {{"tool": "list_files", "arguments": {{"project_name": "my-project"}}, "investigation": true, "dependencies": []}}}}
WITH DEPENDENCY: {{"action": "task", "task": {{"tool": "write_file", "arguments": {{"project_name": "test-project", "file_path": "src/service.js", "content": "brief change description"}}, "dependencies": [{{"tool": "read_file", "arguments": {{"project_name": "test-project", "file_path": "src/service.js"}}}}]}}}}
FINISH: {{"action": "finish", "content": "Description of the final result."}}
FAIL: {{"action": "fail", "reason": "Reason for failure."}}
"checklist_progress" is optional — omit when nothing was completed.

OBJECTIVE:

{objective}

CURRENT CONTEXT:

{context}
"""