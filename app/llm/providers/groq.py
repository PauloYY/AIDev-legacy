from app.config import Config
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        super().__init__(
            base_url=self.BASE_URL,
            api_key=Config.groq_api_key,
            model=Config.groq_model,
        )