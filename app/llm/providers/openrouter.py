from app.config import Config
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        super().__init__(
            base_url=self.BASE_URL,
            api_key=Config.openrouter_api_key,
            model=Config.openrouter_model,
        )