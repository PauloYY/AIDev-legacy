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

    projects_dir = os.getenv("AIDEV_PROJECTS_DIR", "projects")

    log_level = os.getenv("AIDEV_LOG_LEVEL", "INFO")
    log_file = os.getenv("AIDEV_LOG_FILE", "aidev.log")

    max_iterations = int(os.getenv("AIDEV_MAX_ITERATIONS", "50"))

    # Nome -> (api_key, model) usado por Config.available_providers().
    _PROVIDER_FIELDS = {
        "groq": ("groq_api_key", "groq_model"),
        "openrouter": ("openrouter_api_key", "openrouter_model"),
        "cerebras": ("cerebras_api_key", "cerebras_model"),
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
