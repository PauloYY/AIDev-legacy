from app.agent.context.project_summary import ProjectSummary
from app.agent.execution.task import Task
from app.llm.client import LLMClient
from app.llm.models import Message


class ProjectSummaryUpdater:

    def __init__(
        self,
        llm: LLMClient,
        summary: ProjectSummary,
    ):
        self.llm = llm
        self.summary = summary

    def update(
        self,
        objective: str,
        project_name: str,
        task: Task,
        result: str,
    ) -> str:

        current_summary = self.summary.read(
            project_name
        )

        prompt = f"""
Você é responsável por manter o resumo persistente
de um projeto de desenvolvimento.

Atualize o resumo do projeto utilizando as informações
da tarefa que acabou de ser executada.

OBJETIVO:
{objective}

RESUMO ATUAL:
{current_summary}

TASK EXECUTADA:
Tool: {task.tool}

ARGUMENTOS:
{task.arguments}

RESULTADO:
{result}

REGRAS:
- Retorne SOMENTE o conteúdo completo do novo resumo.
- Não retorne JSON.
- Preserve informações importantes do resumo atual.
- Incorpore as novas informações descobertas.
- Não invente informações.
- Remova informações que comprovadamente estejam incorretas.
- O resumo deve representar o estado atual conhecido do projeto.
- Seja conciso.
"""

        response = self.llm.generate(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ]
        )

        updated_summary = response.content.strip()

        if not updated_summary:
            raise ValueError(
                "A LLM retornou um resumo vazio."
            )

        self.summary.write(
            project_name,
            updated_summary,
        )

        return updated_summary