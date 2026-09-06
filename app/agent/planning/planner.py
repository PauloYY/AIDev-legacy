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
        iteration: int | None = None,
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
            ],
            component="Planner",
            iteration=iteration,
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
- "read_file", "list_files" e "find_references" NUNCA podem ser a
  tool de uma task sozinha — são ferramentas de investigação, só
  podem ser usadas dentro de "dependencies". Se você precisa saber
  o conteúdo de um arquivo antes de editá-lo, anexe o read_file
  como dependency da própria task de write_file/run_command; não
  crie uma task só para ler.
- EXCEÇÃO: logo depois que um "run_command" de teste/build falhar,
  você PODE usar read_file/list_files/find_references sozinho como
  task principal, para investigar o que quebrou (ex.: ler o arquivo
  do stack trace) antes de saber qual correção fazer. Fora desse
  caso específico, a regra acima vale normalmente — não abuse da
  exceção encadeando várias investigações soltas seguidas.
- OUTRA EXCEÇÃO: se o seu OBJETIVO for analisar, entender ou
  investigar um projeto existente (sem criar ou modificar código),
  marque a task com "investigation": true. Isso permite usar
  read_file/list_files/find_references como ação principal de forma
  explícita. Exemplo:
  {{"action": "task", "task": {{"tool": "read_file", "arguments": {{...}}, "investigation": true}}}}
  Sem esse campo, a task será rejeitada pelo validador.
- Uma task pode ter várias dependencies ao mesmo tempo — se precisa
  ler mais de um arquivo antes de agir, anexe todos de uma vez em
  vez de fazer isso em tasks separadas.
- Para argumentos de conteúdo extenso (ex.: "content" de
  write_file), NÃO escreva o conteúdo final aqui — descreva
  brevemente o que deve ser feito (ex.: "implementar a classe
  Client com os campos id, nome e status"). O EXECUTOR é quem
  gera o conteúdo completo do arquivo, então repeti-lo aqui
  desperdiça tokens e aumenta o risco de resposta cortada.

PROGRESSO:

- Analise o contexto antes de decidir.
- Considere tudo que já foi executado.
- Não repita uma task que já foi concluída com sucesso.
- Não repita a mesma ferramenta com os mesmos argumentos sem
  uma justificativa concreta.
- Se uma ação falhou, use o erro para decidir o próximo passo.
- Cada nova task deve produzir progresso real.
- Tasks marcadas com "investigation": true são ações de leitura/análise
  pura — não produzem mutação no projeto. Elas não contam como
  estagnação nos mecanismos internos, mas o Planner não deve
  investigar indefinidamente: após coletar informação suficiente,
  tome uma decisão (finish, task de ação, ou fail).
- Antes de escrever um import/require que referencia outro arquivo
  do projeto, use a tool list_symbols nesse arquivo para confirmar o
  nome exato exportado, em vez de adivinhar o nome do arquivo ou do
  símbolo.
- Se o objetivo já estiver concluído, use "finish".
- Use "fail" somente quando não for possível continuar.

CHECKLIST DO OBJETIVO:

- O contexto inclui um CHECKLIST DO OBJETIVO com itens numerados,
  definido uma única vez no início e nunca reescrito.
- Use-o como roteiro: siga a ordem dos itens pendentes em vez de
  redescobrir o que falta fazer a cada iteração.
- Quando a task que você está decidindo agora completar um ou mais
  itens do checklist, inclua o campo opcional
  "checklist_progress": [id, id, ...] na sua decisão (funciona em
  decisões "task", "finish" ou "fail") com os ids concluídos.
- Só marque um item quando ele já foi de fato realizado (ex.: o
  write_file/run_command que o implementa já foi executado com
  sucesso), não quando você está apenas planejando fazê-lo agora.
- O checklist é um guia, não uma camisa de força: se descobrir que
  um item não é mais necessário ou que falta um item novo, siga o
  estado real do projeto — não se prenda ao checklist às custas de
  ignorar um erro real.

CHECKLIST DE ERROS:

- Quando o contexto incluir um CHECKLIST DE ERROS ATUAIS, isso
  significa que o último run_command de teste/build falhou e cada
  item é uma falha concreta extraída automaticamente da saída real.
- Use-o para saber exatamente o que precisa ser corrigido, em vez de
  reler a saída bruta do teste toda hora.
- Esse checklist é automático: não existe campo para marcar item
  como resolvido. Ele some sozinho assim que você rodar o
  teste/build de novo e ele passar; se ainda falhar, é regerado do
  zero com as falhas atuais. Não invente que um item foi corrigido
  sem antes confirmar rodando o teste de novo.
- Quando ele aparecer, priorize corrigir esses erros antes de seguir
  para itens novos do CHECKLIST DO OBJETIVO.
- Enquanto houver qualquer item nesse checklist, "finish" será
  bloqueado — não tente finalizar com falhas de teste/build
  pendentes, corrija e confirme que o teste/build passa primeiro.

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

VALIDAÇÃO ANTES DE FINALIZAR:

- Escrever o código NÃO significa que ele funciona. Não assuma
  que está correto apenas por tê-lo escrito.
- Antes de usar "finish", use "run_command" para executar o
  programa, os testes ou o build (o que fizer sentido para a
  linguagem/framework do projeto) e confirmar que ele realmente
  atende ao objetivo.
- Escolha o comando de acordo com o projeto: "python main.py",
  "python -m pytest -q", "npm test", "npm run build",
  "node app.js", "gcc main.c -o main && ./main",
  "g++ main.cpp -o main && ./main",
  "javac Main.java && java Main", "go run .", "go test ./...",
  "cargo run", "cargo test", "ruby main.rb", "php main.php",
  "mvn test", "gradle test", etc. Use somente comandos cujo
  runtime esteja disponível na imagem de sandbox.
- Para projetos Gradle, use "gradle" diretamente — NUNCA
  "./gradlew" (o wrapper tenta baixar o Gradle pela internet, e
  o sandbox não tem acesso à rede).
- Para projetos Node.js/JavaScript: SEMPRE crie o "package.json"
  ANTES de rodar "npm install" ou "npm test". Nunca rode
  "npm install" num projeto que ainda não tem "package.json" —
  isso falha e pode deixar um "package-lock.json" inconsistente
  para trás, que atrapalha instalações futuras mesmo depois do
  "package.json" correto existir.
- Evite comandos que não terminam sozinhos, como servidores
  (ex.: "npm start", "flask run", "python -m http.server") — eles
  vão estourar o timeout e não servem como validação.
- Se a execução mostrar erro, exceção, saída incorreta ou
  comportamento inesperado, isso NÃO é motivo para "finish" —
  crie uma task para corrigir o problema.
- APÓS UM TESTE/EXECUÇÃO FALHAR, siga este raciocínio antes de
  decidir a próxima ação:
  1. Leia a mensagem de erro com atenção: qual arquivo, função ou
     linha ela aponta? Qual é a causa provável (ex.: nome errado,
     tipo incorreto, lógica invertida, import faltando)?
  2. Se a mensagem já indica claramente o que está errado, corrija
     DIRETO com "write_file" (use "read_file" como dependency da
     própria task de correção se precisar confirmar o conteúdo
     atual antes de editar — não como uma task separada).
  3. NÃO rode o mesmo teste/comando de novo sem antes ter mudado
     algum código — rodar de novo sem mudar nada sempre dá o
     mesmo resultado e não é progresso.
  4. Depois de corrigir, rode o teste de novo para confirmar que
     o problema foi resolvido.
- Para programas interativos (com input()), use o argumento
  "stdin" do "run_command" para simular as entradas do usuário
  e validar os fluxos principais.
- Só use "finish" depois de validar por execução real que o
  objetivo foi atingido.

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
            "content": "breve descrição do que o arquivo deve conter (não o conteúdo completo)"
        }},
        "dependencies": []
    }},
    "checklist_progress": [1]
}}

FORMATO TASK COM INVESTIGAÇÃO (quando o objetivo é analisar/investir
um projeto existente sem modificar nada):

{{
    "action": "task",
    "task": {{
        "tool": "list_files",
        "arguments": {{
            "project_name": "meu-projeto"
        }},
        "investigation": true,
        "dependencies": []
    }}
}}

FORMATO TASK COM DEPENDENCY (quando precisa ver algo antes de agir):

{{
    "action": "task",
    "task": {{
        "tool": "write_file",
        "arguments": {{
            "project_name": "test-project",
            "file_path": "src/service.js",
            "content": "breve descrição da alteração, considerando o conteúdo atual do arquivo"
        }},
        "dependencies": [
            {{
                "tool": "read_file",
                "arguments": {{
                    "project_name": "test-project",
                    "file_path": "src/service.js"
                }}
            }}
        ]
    }},
    "checklist_progress": [1]
}}

FORMATO FINISH:

{{
    "action": "finish",
    "content": "Descrição do resultado final.",
    "checklist_progress": [5]
}}

FORMATO FAIL:

{{
    "action": "fail",
    "reason": "Motivo da falha."
}}

"checklist_progress" é sempre opcional — omita quando a task atual
não concluir nenhum item do checklist.

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