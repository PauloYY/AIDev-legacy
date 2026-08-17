from app.agent.agent import Agent
from app.llm.router import LLMRouter
from app.exceptions import LLMAPIError, LLMRateLimitError
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.providers.groq import GroqProvider
from app.llm.client import LLMClient
from app.tools.filesystem import (
    LIST_FILES_DEFINITION,
    READ_FILE_DEFINITION,
    list_files,
    read_file
)
from app.tools.registry import ToolRegistry


def main():
    router = LLMRouter([GroqProvider(), OpenRouterProvider()])
    llm = LLMClient(router)

    tools = ToolRegistry()

    tools.register(
        "list_files",
        list_files,
        LIST_FILES_DEFINITION,
    )

    tools.register(
        "read_file",
        read_file,
        READ_FILE_DEFINITION,
    )

    agent = Agent(
        llm=llm,
        tools=tools,
    )

    try:

        response = agent.run(
            "Primeiro descubra quais arquivos existem no projeto test-project."
            "Depois leia o main.py."
            "Por fim explique o que o projeto faz."
        )
        print(response)

    except LLMRateLimitError as error:
        print(f"AIDev: limite da LLM atingido.")
        print(error)

    except LLMAPIError as error:
        print(f"AIDev: erro na API da LLM.")
        print(error)
    


if __name__ == "__main__":
    main()