import json

from app.agent.planning.decision import Decision
from app.agent.planning.decision_parser import DecisionParser
from app.llm.client import LLMClient
from app.tools.registry import ToolRegistry
from app.llm.models import Message


class Planner:

    def __init__(
        self,
        llm: LLMClient,
        parser: DecisionParser,
        tools: ToolRegistry,
    ):
        self.llm = llm
        self.parser = parser
        self.tools = tools

    def plan(
        self,
        objective: str,
        context: str = "",
    ) -> Decision:

        prompt = self._build_prompt(
            objective,
            context,
        )

        response = self.llm.generate(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ]
        )

        if response.tool_calls:
            raise ValueError(
                "O Planner tentou executar ferramentas diretamente. "
                "O Planner deve retornar somente um objeto JSON em texto."
            )

        if response.content is None:
            raise ValueError(
                "O Planner não retornou conteúdo. "
                "A resposta deve conter somente um objeto JSON em texto."
            )

        return self.parser.parse(response.content)

    def _build_prompt(
        self,
        objective: str,
        context: str,
    ) -> str:

        return f"""
Você é o PLANNER de um agente autônomo de desenvolvimento.

Sua função é escolher a PRÓXIMA ação necessária para atingir o
objetivo.

REGRAS:

- Retorne SOMENTE JSON válido.
- Nunca execute ferramentas.
- Nunca produza tool calls.
- Use somente as ferramentas disponíveis.
- Não invente ferramentas.
- Uma task representa uma única ação.
- Dependencies são usadas para obter informações antes da task.
- Dependencies podem usar somente ferramentas de análise.

PROGRESSO:

- Analise o contexto antes de decidir.
- Considere tudo que já foi executado.
- Não repita uma task que já foi concluída com sucesso.
- Não repita a mesma ferramenta com os mesmos argumentos sem
  uma justificativa concreta.
- Se uma ação falhou, use o erro para decidir o próximo passo.
- Cada nova task deve produzir progresso real.
- Se o objetivo já estiver concluído, use "finish".
- Use "fail" somente quando não for possível continuar.

COERÊNCIA DO PROJETO:

- Preserve os nomes e interfaces existentes.
- Não invente classes, funções, métodos, atributos ou parâmetros
  que deveriam existir no projeto.
- Se precisar utilizar algo definido em outro arquivo, confirme
  sua definição usando dependencies quando necessário.
- Se precisar alterar uma interface existente, considere seus
  consumidores antes da alteração.
- O estado atual do projeto tem prioridade sobre informações
  antigas do resumo.

TOOLS DISPONÍVEIS:

{self._build_tools_context()}

FORMATO TASK:

{{
    "action": "task",
    "task": {{
        "tool": "write_file",
        "arguments": {{
            "project_name": "test-project",
            "file_path": "result.py",
            "content": "..."
        }},
        "dependencies": []
    }}
}}

FORMATO FINISH:

{{
    "action": "finish",
    "content": "Descrição do resultado final."
}}

FORMATO FAIL:

{{
    "action": "fail",
    "reason": "Motivo da falha."
}}

OBJETIVO:

{objective}

CONTEXTO ATUAL:

{context}

Antes de responder:

1. Analise o objetivo.
2. Analise o que já foi feito.
3. Identifique o que ainda precisa ser feito.
4. Determine se precisa investigar alguma informação.
5. Escolha UMA única próxima ação.

Não repita ações concluídas sem necessidade.

Se não houver mais nada a fazer, escolha "finish".

Retorne SOMENTE o JSON.
"""

    def _build_tools_context(self) -> str:
        return json.dumps(
            self.tools.definitions,
            indent=2,
            ensure_ascii=False,
        )