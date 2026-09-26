import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    # Provedores de LLM: chave e modelo de cada um.
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL")

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")

    cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
    cerebras_model = os.getenv("CEREBRAS_MODEL")

    agnes_api_key = os.getenv("AGNES_API_KEY")
    agnes_model = os.getenv("AGNES_MODEL")

    # Pasta raiz onde os projetos são criados.
    projects_dir = os.getenv("AIDEV_PROJECTS_DIR", "projects")

    # Log: nível e arquivo de saída.
    log_level = os.getenv("AIDEV_LOG_LEVEL", "INFO")
    log_file = os.getenv("AIDEV_LOG_FILE", "aidev.log")

    # Tetos da run: iterações e tokens de saída da LLM.
    max_iterations = int(os.getenv("AIDEV_MAX_ITERATIONS", "50"))

    llm_max_output_tokens = int(os.getenv("AIDEV_LLM_MAX_OUTPUT_TOKENS", "8192"))

    # Espera entre providers quando há rate limit.
    rate_limit_max_wait_rounds = int(os.getenv("AIDEV_RATE_LIMIT_MAX_WAIT_ROUNDS", "3"))
    rate_limit_base_wait_seconds = float(os.getenv("AIDEV_RATE_LIMIT_BASE_WAIT_SECONDS", "10"))
    rate_limit_max_wait_seconds = float(os.getenv("AIDEV_RATE_LIMIT_MAX_WAIT_SECONDS", "90"))

    # Timeout de cada comando executado (segundos).
    execution_timeout_seconds = int(os.getenv("AIDEV_EXECUTION_TIMEOUT_SECONDS", "15"))

    # Otimizações: leituras em paralelo e prompts compactos (1=liga, 0=desliga).
    parallel_tools = os.getenv("AIDEV_PARALLEL_TOOLS", "1") != "0"

    smart_summary = os.getenv("AIDEV_SMART_SUMMARY", "1") != "0"
    compact_context = os.getenv("AIDEV_COMPACT_CONTEXT", "1") != "0"

    compact_executor = os.getenv("AIDEV_COMPACT_EXECUTOR", "1") != "0"

    compact_planner = os.getenv("AIDEV_COMPACT_PLANNER", "1") != "0"

    # Estado da tarefa: contexto, persistência e plano (1=liga, 0=desliga).
    task_state_context = os.getenv("AIDEV_TASK_STATE_CONTEXT", "1") != "0"
    task_state_persist = os.getenv("AIDEV_TASK_STATE_PERSIST", "1") != "0"
    canonical_objective = os.getenv("AIDEV_CANONICAL_OBJECTIVE", "1") != "0"
    pt_ascii_translation = (
        os.getenv("AIDEV_PT_ASCII_TRANSLATION", "1") != "0")
    task_plan_sync = os.getenv("AIDEV_TASK_PLAN_SYNC", "1") != "0"

    # Erros: análise unificada e bloqueio de retest com falha aberta.
    error_analyzer = os.getenv("AIDEV_ERROR_ANALYZER", "1") != "0"
    error_test_gate = os.getenv("AIDEV_ERROR_TEST_GATE", "1") != "0"

    # Sandbox Docker: imagem, limites e rede do container.
    sandbox_mode = os.getenv("AIDEV_SANDBOX_MODE", "docker").lower()
    sandbox_docker_image = os.getenv("AIDEV_SANDBOX_DOCKER_IMAGE", "aidev-sandbox:latest")
    sandbox_memory_limit = os.getenv("AIDEV_SANDBOX_MEMORY_LIMIT", "256m")
    sandbox_cpu_limit = os.getenv("AIDEV_SANDBOX_CPU_LIMIT", "1")
    sandbox_pids_limit = os.getenv("AIDEV_SANDBOX_PIDS_LIMIT", "128")

    sandbox_network_mode = os.getenv("AIDEV_SANDBOX_NETWORK_MODE", "none").lower()
    sandbox_network_name = os.getenv("AIDEV_SANDBOX_NETWORK_NAME", "aidev-sandbox-net")
    sandbox_proxy_container_name = os.getenv(
        "AIDEV_SANDBOX_PROXY_CONTAINER", "aidev-sandbox-proxy"
    )
    sandbox_proxy_image = os.getenv(
        "AIDEV_SANDBOX_PROXY_IMAGE", "aidev-sandbox-proxy:latest"
    )
    sandbox_proxy_port = int(os.getenv("AIDEV_SANDBOX_PROXY_PORT", "3128"))

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
