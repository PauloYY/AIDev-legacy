from app.config import Config
from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class AgnesProvider(OpenAICompatibleProvider):

    BASE_URL = "https://apihub.agnes-ai.com/v1/chat/completions"

    def __init__(self):
        super().__init__(
            base_url=self.BASE_URL,
            api_key=Config.agnes_api_key,
            model=Config.agnes_model,
            name="agnes",
        )
