from app.agent.execution_decision import ExecutionDecision
from app.agent.execution_decision_parser import ExecutionDecisionParser
from app.agent.task import Task
from app.llm.client import LLMClient
from app.llm.models import Message


class TaskDecisionMaker:

    def __init__(
        self,
        llm: LLMClient,
        parser: ExecutionDecisionParser,
    ):
        self.llm = llm
        self.parser = parser

    def decide(
        self,
        objective: str,
        task: Task,
        context: str,
    ) -> ExecutionDecision:

        prompt = f"""
Você é o executor de uma tarefa de desenvolvimento.

Sua função é decidir como executar a task fornecida,
utilizando as informações disponíveis no contexto.

OBJETIVO:
{objective}

TASK:
Tool: {task.tool}

ARGUMENTOS:
{task.arguments}

CONTEXTO DAS DEPENDÊNCIAS:
{context}

REGRAS:
- Retorne SOMENTE JSON válido.
- Use exclusivamente a tool indicada na task.
- Não crie novas tasks.
- Não crie dependencies.
- Não altere a finalidade da task.
- Os argumentos devem ser válidos para a tool.
- Para write_file, forneça o conteúdo completo do arquivo.
- Para read_file, mantenha os argumentos necessários para leitura.

FORMATO:
{{
    "tool": "{task.tool}",
    "arguments": {{
        ...
    }}
}}

Retorne SOMENTE o JSON.
"""

        response = self.llm.generate(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ]
        )

        return self.parser.parse(response.content)