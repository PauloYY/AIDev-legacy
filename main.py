from app.agent.decision_parser import DecisionParser
from app.agent.planner import Planner
from app.agent.project_analyzer import ProjectAnalyzer
from app.agent.project_context import ProjectContext
from app.agent.project_summary import ProjectSummary
from app.agent.runner import Runner
from app.agent.task_context_builder import TaskContextBuilder
from app.agent.task_decision_maker import TaskDecisionMaker
from app.agent.execution_decision_parser import ExecutionDecisionParser
from app.agent.project_summary_updater import ProjectSummaryUpdater

from app.exceptions import LLMAPIError, LLMRateLimitError

from app.llm.client import LLMClient
from app.llm.providers.groq import GroqProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.router import LLMRouter

from app.tools.registry import ToolRegistry

from app.cli import handle_agent_event


PROJECT_NAME = "guess_game"


def main():
    tools = ToolRegistry()
    tools.load_defaults()

    router = LLMRouter([
        GroqProvider(),
        OpenRouterProvider(),
    ])

    llm = LLMClient(router)

    summary = ProjectSummary()

    summary_updater = ProjectSummaryUpdater(
        summary=summary,
        llm=llm
    )

    analyzer = ProjectAnalyzer(
        llm=llm,
    )

    project_context = ProjectContext(
        summary=summary,
        analyzer=analyzer,
        tools=tools,
    )

    planner = Planner(
        llm=llm,
        parser=DecisionParser(),
        tools=tools,
    )

    task_context_builder = TaskContextBuilder()

    task_decision_maker = TaskDecisionMaker(
        llm=llm,
        parser=ExecutionDecisionParser(),
    )

    runner = Runner(
        planner=planner,
        task_decision_maker=task_decision_maker,
        task_context_builder=task_context_builder,
        tools=tools,
        project_context=project_context,
        project_summary_updater=summary_updater,
    )

    try:
        response = runner.run(
            objective=(
                """
                Crie um jogo simples de adivinhação para ser executado no terminal.

                O programa deve:

                * gerar aleatoriamente um número entre 1 e 100;
                * pedir ao jogador que tente adivinhar o número;
                * informar se o palpite é maior ou menor que o número secreto;
                * continuar pedindo palpites até que o jogador acerte;
                * informar ao jogador quantas tentativas foram necessárias.

                Use a pasta `guess_game` para o projeto.

                Mantenha o projeto simples e organizado.

                """
            ),
            project_name=PROJECT_NAME,
        )

        print("\nResposta:")
        print(response)

    except LLMRateLimitError as error:
        print("\n* Limite de requisições da LLM atingido.")
        print(f"   {error}")

    except LLMAPIError as error:
        print("\n* Erro na API da LLM.")
        print(f"   {error}")


if __name__ == "__main__":
    main()