import logging
import sys

from app.agent.context.project_analyzer import ProjectAnalyzer
from app.agent.context.project_context import ProjectContext
from app.agent.context.project_summary import ProjectSummary
from app.agent.context.project_summary_updater import ProjectSummaryUpdater
from app.agent.context.task_context_builder import TaskContextBuilder
from app.agent.context.operational_memory import OperationalMemory
from app.agent.context.checklist import ProjectChecklist
from app.agent.context.error_checklist import ErrorChecklist
from app.agent.execution.execution_decision_parser import ExecutionDecisionParser
from app.agent.execution.task_decision_maker import TaskDecisionMaker
from app.agent.execution.validator import TaskValidator
from app.agent.planning.decision_parser import DecisionParser
from app.agent.planning.planner import Planner
from app.agent.runner import Runner

from app.cli.args import parse_args
from app.cli.events import handle_agent_event

from app.config import Config
from app.exceptions import ConfigurationError, LLMAPIError, LLMConnectionError, LLMRateLimitError
from app.logging_config import setup_logging

from app.llm.client import LLMClient
from app.llm.providers.base import LLMProvider
from app.llm.providers.agnes import AgnesProvider
from app.llm.providers.groq import GroqProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.router import LLMRouter

from app.tools.execution import sandbox
from app.tools.registry import ToolRegistry

from app.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "agnes": AgnesProvider,
}


def build_providers() -> list[LLMProvider]:
    """Instancia somente os providers com API key + modelo configurados.

    Antes, os 3 providers eram sempre instanciados incondicionalmente —
    um provider sem credenciais só falhava (com um erro HTTP genérico)
    quando o LLMRouter tentava usá-lo. Agora a ausência de configuração
    é detectada e reportada claramente antes de qualquer chamada de rede.
    """

    available = Config.available_providers()

    if not available:
        missing = Config.missing_providers()

        details = "\n".join(
            f"  - {name}: faltam {', '.join(variables)}"
            for name, variables in missing.items()
        )

        raise ConfigurationError(
            "Nenhum provider de LLM está configurado corretamente.\n"
            "Configure ao menos um dos seguintes no seu .env:\n"
            f"{details}\n\n"
            "Veja .env.example para o formato esperado."
        )

    missing = Config.missing_providers()

    for name, variables in missing.items():
        logger.warning(
            "Provider '%s' incompleto (faltam: %s) — será ignorado.",
            name,
            ", ".join(variables),
        )

    return [PROVIDER_CLASSES[name]() for name in available]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    setup_logging(level=args.log_level, log_file=args.log_file)

    try:
        providers = build_providers()
    except ConfigurationError as error:
        print(f"\n* Erro de configuração:\n{error}")
        return 1

    logger.info(
        "Providers configurados: %s",
        ", ".join(getattr(p, "name", type(p).__name__) for p in providers),
    )

    if Config.sandbox_mode == "docker":
        available, reason = sandbox.sandbox_available()

        if not available:
            print(f"\n* Sandbox Docker indisponível:\n{reason}")
            return 1

        logger.info(
            "Sandbox Docker OK (imagem: %s).",
            Config.sandbox_docker_image,
        )

    tools = ToolRegistry()
    tools.load_defaults()

    router = LLMRouter(
        providers,
        max_wait_rounds=Config.rate_limit_max_wait_rounds,
        base_wait_seconds=Config.rate_limit_base_wait_seconds,
        max_wait_seconds=Config.rate_limit_max_wait_seconds,
    )
    llm = LLMClient(router)

    summary = ProjectSummary()

    summary_updater = ProjectSummaryUpdater(
        summary=summary,
        llm=llm,
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
        parser=DecisionParser(tools=tools),
        tools=tools,
    )

    task_context_builder = TaskContextBuilder()

    task_decision_maker = TaskDecisionMaker(
        llm=llm,
        parser=ExecutionDecisionParser(),
    )

    validator = TaskValidator(tools)
    operational_memory = OperationalMemory(tools)
    checklist = ProjectChecklist(llm)
    error_checklist = ErrorChecklist(llm)

    runner = Runner(
        planner=planner,
        task_decision_maker=task_decision_maker,
        task_context_builder=task_context_builder,
        tools=tools,
        project_context=project_context,
        project_summary_updater=summary_updater,
        validator=validator,
        operational_memory=operational_memory,
        checklist=checklist,
        error_checklist=error_checklist,
        on_event=handle_agent_event,
        max_iterations=args.max_iterations,
    )

    if args.is_default_demo:
        print(
            "(nenhum --objective informado — executando objetivo de "
            "demonstração; use --help para ver as opções)\n"
        )

    try:
        response = runner.run(
            objective=args.objective,
            project_name=args.project_name,
        )

        print("\nResposta:")
        print(response)

        return 0

    except LLMRateLimitError as error:
        print("\n* Limite de requisições da LLM atingido.")
        print(f"   {error}")
        return 1

    except LLMConnectionError as error:
        print("\n* Falha de conexão com a LLM.")
        print(f"   {error}")
        return 1

    except LLMAPIError as error:
        print("\n* Erro na API da LLM.")
        print(f"   {error}")
        return 1

    except RuntimeError as error:
        print(f"\n* {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())