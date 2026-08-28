# AIDev

AIDev é um agente autônomo de desenvolvimento de software. Você descreve um
objetivo em linguagem natural, e o agente planeja e executa iterativamente
as tarefas necessárias — usando ferramentas de leitura, escrita e busca em
arquivos — até completar o objetivo, dentro de um diretório de projeto
isolado.

Suporta múltiplos providers de LLM compatíveis com a API da OpenAI (Groq,
OpenRouter, Cerebras), com fallback automático entre eles em caso de rate
limit ou falha transitória.

## Arquitetura

O agente opera em ciclos. A cada iteração:

```
                        ┌───────────────────────┐
                        │      Runner.run()      │
                        └───────────┬────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │   Planner: qual a próxima     │
                     │   ação para atingir o         │
                     │   objetivo? (task/finish/fail) │
                     └──────────────┬───────────────┘
                                    │ task
                     ┌──────────────▼───────────────┐
                     │   TaskValidator: a tool e os   │
                     │   argumentos são válidos?      │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  Dependencies: resolve infos    │
                     │  necessárias (tools de análise) │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  TaskDecisionMaker: decide os   │
                     │  argumentos finais de execução  │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │   ToolRegistry.execute(...)     │
                     │   (list_files, read_file,       │
                     │    write_file, find_references) │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  ProjectSummaryUpdater: atualiza│
                     │  o resumo persistente do projeto│
                     │  (.aidev/summary.md)            │
                     └───────────────────────────────┘
```

Separação de responsabilidades por pacote:

- **`app/agent/planning/`** — o Planner decide a próxima ação de alto
  nível (task / finish / fail).
- **`app/agent/execution/`** — valida a task (`TaskValidator`) e decide
  os argumentos concretos de execução (`TaskDecisionMaker`).
- **`app/agent/context/`** — mantém o entendimento do projeto: análise
  inicial (`ProjectAnalyzer`) e resumo persistente
  (`ProjectSummary` / `ProjectSummaryUpdater`).
- **`app/tools/`** — ferramentas que o agente pode executar. Cada tool
  é classificada como `ANALYSIS` (somente leitura, pode ser usada como
  dependency) ou `EXECUTION` (modifica o projeto).
- **`app/llm/`** — abstração de LLM: `LLMClient` → `LLMRouter` →
  providers concretos (Groq/OpenRouter/Cerebras), todos implementando a
  mesma interface `LLMProvider`.
- **`app/cli/`** — interface de linha de comando (`args.py`) e feedback
  visual de progresso no terminal (`events.py`).

Toda tool de filesystem valida que o caminho resolvido permanece dentro do
diretório do projeto correspondente (proteção contra path traversal).

## Instalação

Requer Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e configure ao menos um provider de LLM:

```bash
cp .env.example .env
```

## Uso

```bash
python main.py --project meu_projeto --objective "Descreva aqui o que o agente deve construir."
```

Ou a partir de um arquivo de texto com o objetivo:

```bash
python main.py --project meu_projeto --objective-file objetivo.txt
```

Executar sem argumentos roda um objetivo de demonstração (um gerenciador de
hábitos em terminal):

```bash
python main.py
```

Ver todas as opções (nível de log, arquivo de log, limite de iterações):

```bash
python main.py --help
```

Os projetos gerados ficam em `projects/<nome-do-projeto>/` (configurável via
`AIDEV_PROJECTS_DIR`). O resumo persistente de cada projeto fica em
`projects/<nome-do-projeto>/.aidev/summary.md`.

## Configuração (.env)

| Variável | Descrição |
|---|---|
| `GROQ_API_KEY` / `GROQ_MODEL` | Credenciais do provider Groq |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | Credenciais do provider OpenRouter |
| `CEREBRAS_API_KEY` / `CEREBRAS_MODEL` | Credenciais do provider Cerebras |
| `AIDEV_PROJECTS_DIR` | Diretório raiz dos projetos (padrão: `projects`) |
| `AIDEV_LOG_LEVEL` | Nível de log interno (padrão: `INFO`) |
| `AIDEV_LOG_FILE` | Arquivo de log interno, ou `none` para desativar (padrão: `aidev.log`) |
| `AIDEV_MAX_ITERATIONS` | Limite global de iterações do agente (padrão: `50`) |

Somente providers com **API key e modelo** configurados são usados. Se
nenhum estiver completo, o programa informa exatamente o que falta antes de
fazer qualquer chamada de rede.

## Testes

```bash
pytest
```

Cobre os parsers de decisão, as tools de filesystem (incluindo as proteções
contra path traversal), a lógica de fallback do `LLMRouter` e o
`TaskValidator`.

## Limitações conhecidas

- O agente não executa código gerado — apenas escreve/lê arquivos. Validar
  o resultado (rodar testes, executar o programa) é responsabilidade de
  quem usa a ferramenta.
- Sem suporte a chamadas assíncronas/paralelas: cada chamada de LLM é
  síncrona e bloqueante.
- `AIDEV_MAX_ITERATIONS` evita loops infinitos, mas não garante que o
  objetivo será concluído dentro do limite — objetivos muito complexos
  podem precisar de um limite maior.
