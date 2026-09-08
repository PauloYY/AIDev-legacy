from app.agent.context.project_analyzer import ProjectAnalyzer
from app.agent.context.project_summary import ProjectSummary
from app.tools.config import get_projects_dir
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

        self._ensure_project_directory(project_name)

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

    def _ensure_project_directory(self, project_name: str) -> None:
        projects_dir = get_projects_dir()
        project_path = (projects_dir / project_name).resolve()

        if not project_path.is_relative_to(projects_dir):
            raise PermissionError(
                "Access outside the projects directory is not allowed."
            )

        project_path.mkdir(parents=True, exist_ok=True)