import os

from dotenv import load_dotenv

load_dotenv()

class Config:
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model = os.getenv("LLM_MODEL")