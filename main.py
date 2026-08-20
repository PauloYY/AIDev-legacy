from app.agent.agent import Agent
from app.llm.client import LLMClient
from app.tools.registry import ToolRegistry
from app.llm.router import LLMRouter
from app.exceptions import LLMAPIError, LLMRateLimitError

from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.providers.groq import GroqProvider

from app.cli import handle_agent_event

def main():
    tools = ToolRegistry()
    tools.load_defaults()

    router = LLMRouter([
        GroqProvider(),
        OpenRouterProvider(),
    ])

    llm = LLMClient(router)

    agent = Agent(
        llm=llm,
        tools=tools,
        on_event=handle_agent_event,
    )

    try:
        response = agent.run(
            "No projeto test-project na pasta src crie uma função python que calcule algo que faça sentido com os outros arquivos."
        )

        print("\nResposta:")
        print(response)

    except LLMRateLimitError as error:
        print(f"\n* Limite de requisições da LLM atingido.")
        print(f"   {error}")

    except LLMAPIError as error:
        print(f"\n* Erro na API da LLM.")
        print(f"   {error}")


if __name__ == "__main__":
    main()