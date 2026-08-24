from app.agent.decision import DecisionAction
from app.agent.planner import Planner
from app.agent.task_context_builder import TaskContextBuilder
from app.agent.task_decision_maker import TaskDecisionMaker
from app.agent.project_context import ProjectContext
from app.agent.project_summary_updater import ProjectSummaryUpdater
from app.tools.registry import ToolRegistry


class Runner:

    def __init__(
        self,
        planner: Planner,
        task_decision_maker: TaskDecisionMaker,
        task_context_builder: TaskContextBuilder,
        tools: ToolRegistry,
        project_context: ProjectContext,
        project_summary_updater: ProjectSummaryUpdater,
    ):
        self.planner = planner
        self.task_decision_maker = task_decision_maker
        self.task_context_builder = task_context_builder
        self.tools = tools
        self.project_context = project_context
        self.project_summary_updater = project_summary_updater

    def run(
        self,
        objective: str,
        project_name: str,
        context: str = "",
    ):
        summary = self.project_context.initialize(
            project_name
        )

        context = (
            f"RESUMO DO PROJETO:\n"
            f"{summary}\n\n"
            f"{context}"
        )

        while True:

            decision = self.planner.plan(
                objective=objective,
                context=context,
            )

            if decision.action == DecisionAction.FINISH:
                return decision.content

            if decision.action == DecisionAction.FAIL:
                raise RuntimeError(decision.reason)

            if decision.action != DecisionAction.TASK:
                raise ValueError(
                    f"Ação desconhecida: {decision.action}"
                )

            task = decision.task

            dependency_results = []

            for dependency in task.dependencies:
                try:
                    result = self.tools.execute(
                        dependency.tool,
                        dependency.arguments,
                    )
                except Exception as exc:
                    result = (
                        f"ERRO: {type(exc).__name__}: {exc}"
                    )

                dependency_results.append(result)

            task_context = self.task_context_builder.build(
                task,
                dependency_results,
            )

            execution = self.task_decision_maker.decide(
                objective=objective,
                task=task,
                context=task_context,
            )

            if execution.tool != task.tool:
                raise ValueError(
                    "A LLM tentou executar uma tool diferente da tool definida na task."
                )

            try:
                result = self.tools.execute(
                    execution.tool,
                    execution.arguments,
                )
            except Exception as exc:
                result = (
                    f"ERRO: {type(exc).__name__}: {exc}"
                )

            summary = self.project_summary_updater.update(
                objective=objective,
                project_name=project_name,
                task=task,
                result=result,
            )

            context = (
                f"RESUMO DO PROJETO:\n"
                f"{summary}\n\n"
                f"{task_context}\n\n"
                f"RESULTADO DA EXECUÇÃO:\n"
                f"{result}"
            )