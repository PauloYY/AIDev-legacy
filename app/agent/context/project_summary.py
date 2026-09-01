from pathlib import Path

from app.tools.config import get_projects_dir


class ProjectSummary:
    DIRECTORY = ".aidev"
    FILE_NAME = "summary.md"

    def __init__(self):
        self._cache: dict[str, str] = {}

    def exists(self, project_name: str) -> bool:
        if project_name in self._cache:
            return True

        return self._path(project_name).exists()

    def read(self, project_name: str) -> str:
        if project_name in self._cache:
            return self._cache[project_name]

        path = self._path(project_name)

        if not path.exists():
            return ""

        content = path.read_text(encoding="utf-8")
        self._cache[project_name] = content

        return content

    def write(
        self,
        project_name: str,
        content: str,
    ) -> None:
        path = self._path(project_name)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        self._cache[project_name] = content

    def _path(self, project_name: str) -> Path:
        projects_dir = get_projects_dir()
        project_path = (
            projects_dir / project_name
        ).resolve()

        if not project_path.is_relative_to(projects_dir):
            raise PermissionError(
                "Acesso fora do diretório de projetos não permitido."
            )

        return (
            project_path
            / self.DIRECTORY
            / self.FILE_NAME
        )
