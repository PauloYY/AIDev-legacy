import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL")

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")

    cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
    cerebras_model = os.getenv("CEREBRAS_MODEL")

    agnes_api_key = os.getenv("AGNES_API_KEY")
    agnes_model = os.getenv("AGNES_MODEL")

    projects_dir = os.getenv("AIDEV_PROJECTS_DIR", "projects")

    log_level = os.getenv("AIDEV_LOG_LEVEL", "INFO")
    log_file = os.getenv("AIDEV_LOG_FILE", "aidev.log")

    max_iterations = int(os.getenv("AIDEV_MAX_ITERATIONS", "50"))

    llm_max_output_tokens = int(os.getenv("AIDEV_LLM_MAX_OUTPUT_TOKENS", "8192"))

    rate_limit_max_wait_rounds = int(os.getenv("AIDEV_RATE_LIMIT_MAX_WAIT_ROUNDS", "3"))
    rate_limit_base_wait_seconds = float(os.getenv("AIDEV_RATE_LIMIT_BASE_WAIT_SECONDS", "10"))
    rate_limit_max_wait_seconds = float(os.getenv("AIDEV_RATE_LIMIT_MAX_WAIT_SECONDS", "90"))

    execution_timeout_seconds = int(os.getenv("AIDEV_EXECUTION_TIMEOUT_SECONDS", "15"))

    # Paralelização de tools puramente observadoras (Etapa 3): "1"
    # (padrão) permite executar dependencies/gather independentes em
    # paralelo; "0" força o caminho sequencial legado (reversível).
    parallel_tools = os.getenv("AIDEV_PARALLEL_TOOLS", "1") != "0"

    # Etapa 4 — redução de chamadas/contexto (reversíveis p/ benchmark):
    # - AIDEV_SMART_SUMMARY=1 pula o Summary Updater após operações de
    #   leitura pura (não mudam o estado; descobertas seguem inline no
    #   próximo contexto + histórico determinístico).
    # - AIDEV_COMPACT_CONTEXT=1 limita resultados gigantes de
    #   dependencies no contexto do Executor e usa listagem estrutural
    #   de arquivos (só reenvia quando muda).
    smart_summary = os.getenv("AIDEV_SMART_SUMMARY", "1") != "0"
    compact_context = os.getenv("AIDEV_COMPACT_CONTEXT", "1") != "0"

    # Etapa 6 — eficiência do Executor + SummaryUpdater (reversível):
    # - AIDEV_COMPACT_EXECUTOR=1 usa template compacto no Executor,
    #   limita cada resultado de dependency a 2000 chars no contexto do
    #   Executor (Etapa 4 usava 4000; arquivos pequenos seguem
    #   byte-idênticos) e estende os skips do SummaryUpdater para
    #   operações comprovadamente sem mudança de estado (check_project
    #   e run_command não-mutante e não-teste/build — o veredito segue
    #   inline no próximo contexto + histórico + error checklist).
    # - AIDEV_COMPACT_EXECUTOR=0 restaura o comportamento da Etapa 5.
    #   A semântica de AIDEV_SMART_SUMMARY (Etapa 4) fica congelada.
    compact_executor = os.getenv("AIDEV_COMPACT_EXECUTOR", "1") != "0"

    # Etapa 5 — compactação inteligente do contexto do Planner
    # (reversível p/ benchmark):
    # - AIDEV_COMPACT_PLANNER=1 usa template estático compacto +
    #   schemas compactos no Planner, janela de histórico recente com
    #   preservação de falhas, e resultados compactos que preservam o
    #   veredito (STATUS/exit code) para o Planner. O error checklist
    #   continua gerado do resultado INTEGRAL (ver Etapa 6 para o
    #   contexto do Executor).
    # - AIDEV_COMPACT_PLANNER=0 restaura byte a byte o comportamento
    #   anterior (prompt completo legado).
    compact_planner = os.getenv("AIDEV_COMPACT_PLANNER", "1") != "0"

    # Fase 4 (integração) — TaskState como fonte estruturada (tudo
    # reversível; 0 = comportamento anterior à integração):
    # - AIDEV_TASK_STATE_CONTEXT=1 injeta TASK STATE (render_compact,
    #   sem conteúdos integrais) no contexto do Planner e do Executor.
    # - AIDEV_TASK_STATE_PERSIST=1 persiste .aidev/task_state.json
    #   (atômico) e tenta recuperar na mesma tarefa (task_id).
    # - AIDEV_CANONICAL_OBJECTIVE=1 faz o Planner/Executor/checklist/
    #   summary/verificação usar o objective canônico (EN); o original
    #   segue preservado no TaskState + trace.
    # - AIDEV_PT_ASCII_TRANSLATION=1 traduz PT sem acentos com forte
    #   evidência (verbos + palavras funcionais); 0 = só com acentos.
    # - AIDEV_TASK_PLAN_SYNC=1 espelha o checklist em TaskState.plan
    #   (id/status/ref, sem copiar descrições).
    task_state_context = os.getenv("AIDEV_TASK_STATE_CONTEXT", "1") != "0"
    task_state_persist = os.getenv("AIDEV_TASK_STATE_PERSIST", "1") != "0"
    canonical_objective = os.getenv("AIDEV_CANONICAL_OBJECTIVE", "1") != "0"
    pt_ascii_translation = (
        os.getenv("AIDEV_PT_ASCII_TRANSLATION", "1") != "0")
    task_plan_sync = os.getenv("AIDEV_TASK_PLAN_SYNC", "1") != "0"

    sandbox_mode = os.getenv("AIDEV_SANDBOX_MODE", "docker").lower()
    sandbox_docker_image = os.getenv("AIDEV_SANDBOX_DOCKER_IMAGE", "aidev-sandbox:latest")
    sandbox_memory_limit = os.getenv("AIDEV_SANDBOX_MEMORY_LIMIT", "256m")
    sandbox_cpu_limit = os.getenv("AIDEV_SANDBOX_CPU_LIMIT", "1")
    sandbox_pids_limit = os.getenv("AIDEV_SANDBOX_PIDS_LIMIT", "128")

    # Rede do sandbox: "none" (padrão, sem nenhum acesso à rede) ou
    # "restricted" (acesso somente através de um proxy com allowlist de
    # domínios — ver docker/squid/). Nunca use rede totalmente aberta
    # aqui; se precisar de outro modo, adicione um allowlist novo em
    # vez de remover a restrição.
    sandbox_network_mode = os.getenv("AIDEV_SANDBOX_NETWORK_MODE", "none").lower()
    sandbox_network_name = os.getenv("AIDEV_SANDBOX_NETWORK_NAME", "aidev-sandbox-net")
    sandbox_proxy_container_name = os.getenv(
        "AIDEV_SANDBOX_PROXY_CONTAINER", "aidev-sandbox-proxy"
    )
    sandbox_proxy_image = os.getenv(
        "AIDEV_SANDBOX_PROXY_IMAGE", "aidev-sandbox-proxy:latest"
    )
    sandbox_proxy_port = int(os.getenv("AIDEV_SANDBOX_PROXY_PORT", "3128"))

    # Nome -> (api_key, model) usado por Config.available_providers().
    _PROVIDER_FIELDS = {
        "groq": ("groq_api_key", "groq_model"),
        "openrouter": ("openrouter_api_key", "openrouter_model"),
        "cerebras": ("cerebras_api_key", "cerebras_model"),
        "agnes": ("agnes_api_key", "agnes_model"),
    }

    @classmethod
    def available_providers(cls) -> list[str]:
        """Retorna os nomes dos providers com API key E modelo configurados."""

        available = []

        for name, (key_field, model_field) in cls._PROVIDER_FIELDS.items():
            if getattr(cls, key_field) and getattr(cls, model_field):
                available.append(name)

        return available

    @classmethod
    def missing_providers(cls) -> dict[str, list[str]]:
        """Retorna, para cada provider incompleto, quais variáveis faltam."""

        missing: dict[str, list[str]] = {}

        for name, (key_field, model_field) in cls._PROVIDER_FIELDS.items():
            missing_vars = []

            if not getattr(cls, key_field):
                missing_vars.append(key_field.upper())

            if not getattr(cls, model_field):
                missing_vars.append(model_field.upper())

            if missing_vars:
                missing[name] = missing_vars

        return missing