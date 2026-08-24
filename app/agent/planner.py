import json

from app.agent.decision import Decision
from app.agent.decision_parser import DecisionParser
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
                "A LLM tentou executar uma ferramenta diretamente. "
                "O Planner deve retornar somente JSON."
            )

        if response.content is None:
            raise ValueError(
                "A LLM não retornou conteúdo para o Planner."
            )

        return self.parser.parse(response.content)

    def _build_prompt(
        self,
        objective: str,
        context: str,
    ) -> str:
        return f"""
    Você é o planejador de execução de um agente de desenvolvimento.

    Sua função é analisar o objetivo e o contexto atual e decidir
    qual deve ser o próximo passo da execução.

    IMPORTANTE:
    - Você está sendo usado como PLANNER.
    - NÃO execute ferramentas.
    - NÃO produza tool calls.
    - Você pode referenciar ferramentas no JSON da task e das
    dependencies.
    - Sua resposta será processada por um parser JSON.
    - Sua resposta deve conter SOMENTE um objeto JSON em texto.

    REGRAS GERAIS:
    - Retorne SOMENTE JSON válido.
    - Você não executa ferramentas.
    - Uma task representa uma única ação de execução.
    - Uma task pode possuir zero ou mais dependencies.
    - Dependencies servem exclusivamente para obter informações
    necessárias para executar a task.
    - Dependencies devem utilizar apenas ferramentas de análise.
    - Ferramentas de execução não podem ser usadas como dependencies.
    - Use somente as ferramentas disponíveis.
    - Não invente ferramentas.
    - Os argumentos devem seguir exatamente as definições das ferramentas.

    SOBRE O CONTEXTO E O RESUMO:
    - O resumo do projeto é apenas uma fonte de contexto e memória.
    - O resumo pode estar desatualizado.
    - O resumo NÃO deve ser tratado como fonte de verdade sobre
    o estado atual dos arquivos.
    - Quando uma decisão depender do estado atual do projeto,
    confirme essa informação utilizando uma dependency.
    - Informações obtidas por dependencies durante a execução atual
    devem ter prioridade sobre informações antigas presentes no resumo.
    - Não assuma que um arquivo continua existindo apenas porque
    o resumo afirma que ele existe.
    - Não assuma que o conteúdo de um arquivo permanece igual apenas
    porque o resumo descreve esse conteúdo.

    SOBRE VERIFICAÇÃO DE ARQUIVOS:
    - Para verificar se um arquivo existe atualmente, use list_files.
    - NUNCA use read_file apenas para verificar se um arquivo existe.
    - Use read_file para obter o conteúdo de um arquivo que já foi
    confirmado como existente.
    - Se list_files mostrar que um arquivo não existe, considere isso
    como evidência de que o arquivo não existe atualmente.
    - Se precisar verificar a existência e o conteúdo de um arquivo,
    primeiro use list_files e depois read_file somente se o arquivo
    existir.

    SOBRE DEPENDENCIES:
    - Use dependencies quando precisar confirmar informações antes
    de executar a task.
    - Uma dependency pode ser necessária mesmo quando o resumo
    aparentemente possui a informação, caso essa informação possa
    ter mudado.
    - Não use dependencies desnecessárias quando o contexto atual
    já fornecer uma informação confiável e suficiente.
    - Pode haver múltiplas dependencies na mesma task.
    - As dependencies são executadas antes da task pai.
    - Cada dependency deve utilizar uma ferramenta adequada ao tipo
    de informação que precisa ser obtida.

    SOBRE EXECUÇÃO:
    - A task pai deve representar a ação que realmente será executada.
    - Não use uma ferramenta de análise como task pai quando a intenção
    for modificar o projeto.
    - Não repita uma ação que já foi concluída e confirmada pelo
    contexto atual.
    - Se a informação necessária estiver desatualizada ou não puder
    ser confirmada, obtenha a informação novamente através de uma
    dependency.
    - Se uma execução retornar um erro, trate o erro como informação
    sobre o estado atual do projeto.
    - Analise o erro e, se possível, crie uma nova task para corrigir
    o problema.
    - Não considere automaticamente um erro de ferramenta como falha
    definitiva da execução.
    - Use "fail" somente quando não for possível continuar ou se
    recuperar do erro.
    - Se não houver mais nada a executar, use "finish".

    TOOLS DISPONÍVEIS:
    {self._build_tools_context()}

    FORMATO PARA TASK:
    {{
        "action": "task",
        "task": {{
            "tool": "write_file",
            "arguments": {{
                "project_name": "test-project",
                "file_path": "result.py",
                "content": "..."
            }},
            "dependencies": [
                {{
                    "tool": "list_files",
                    "arguments": {{
                        "project_name": "test-project"
                    }}
                }}
            ]
        }}
    }}

    FORMATO PARA FINISH:
    {{
        "action": "finish",
        "content": "Descrição do resultado final."
    }}

    FORMATO PARA FAIL:
    {{
        "action": "fail",
        "reason": "Motivo da falha."
    }}

    OBJETIVO:
    {objective}

    CONTEXTO ATUAL:
    {context}

    Analise o estado atual e escolha apenas uma ação.

    Retorne SOMENTE o JSON.
    """
    def _build_tools_context(self) -> str:
        return json.dumps(
            self.tools.definitions,
            indent=2,
            ensure_ascii=False,
        )