from app.agent.agent import Agent
from app.llm.router import LLMRouter

from app.exceptions import LLMAPIError, LLMRateLimitError

from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.providers.groq import GroqProvider

from app.llm.client import LLMClient
from app.tools.filesystem import (
    LIST_FILES_DEFINITION,
    READ_FILE_DEFINITION,
    WRITE_FILE_DEFINITION,
    list_files,
    read_file,
    write_file
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

    tools.register(
        "write_file",
        write_file,
        WRITE_FILE_DEFINITION
    )

    agent = Agent(
        llm=llm,
        tools=tools,
    )

    try:

        response = agent.run(
            "No projeto test-project melhore o código do arquivo python que faz a média de três valores"
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