from app.agent.agent import Agent
from app.llm.client import LLMClient
from app.tools.filesystem import (
    LIST_FILES_DEFINITION,
    list_files,
)
from app.tools.registry import ToolRegistry


def main():
    llm = LLMClient()

    tools = ToolRegistry()

    tools.register(
        "list_files",
        list_files,
        LIST_FILES_DEFINITION,
    )

    agent = Agent(
        llm=llm,
        tools=tools,
    )

    response = agent.run(
        "Descubra quais arquivos existem no "
        "projeto test-project e me diga quais são."
    )

    print(response)


if __name__ == "__main__":
    main()