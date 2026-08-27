from app.agent.planning.decision_parser import DecisionParser
from app.agent.planning.planner import Planner
from app.agent.context.project_analyzer import ProjectAnalyzer
from app.agent.context.project_context import ProjectContext
from app.agent.context.project_summary import ProjectSummary
from app.agent.runner import Runner
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.execution.task_decision_maker import TaskDecisionMaker
from app.agent.execution.execution_decision_parser import ExecutionDecisionParser
from app.agent.context.project_summary_updater import ProjectSummaryUpdater

from app.exceptions import LLMAPIError, LLMRateLimitError

from app.llm.client import LLMClient
from app.llm.providers.groq import GroqProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.router import LLMRouter

from app.tools.registry import ToolRegistry

from app.cli import handle_agent_event


PROJECT_NAME = "gerador"
prompt = """"
Crie um gerador de senhas automático.
Use a pasta gerador.
Faça de forma simples.

"""

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
        on_event=handle_agent_event,
    )

    try:
        response = runner.run(
            objective=prompt,
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