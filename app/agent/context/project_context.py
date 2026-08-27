from app.agent.context.project_analyzer import ProjectAnalyzer
from app.agent.context.project_summary import ProjectSummary
from app.tools.registry import ToolRegistry


class ProjectContext:
    def __init__(
        self,
        summary: ProjectSummary,
        analyzer: ProjectAnalyzer,
        tools: ToolRegistry,
    ):
        self.summary = summary
        self.analyzer = analyzer
        self.tools = tools

    def initialize(
        self,
        project_name: str,
    ) -> str:
        if self.summary.exists(project_name):
            return self.summary.read(project_name)

        files = self.tools.execute(
            "list_files",
            {
                "project_name": project_name,
            },
        )

        content = self.analyzer.analyze(
            project_name,
            files,
        )

        self.summary.write(
            project_name,
            content,
        )

        return content