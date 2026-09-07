# AIDev

Agente autônomo de desenvolvimento de software: você descreve um objetivo em linguagem natural e o AIDev planeja, executa, testa e corrige iterativamente até concluí-lo — dentro de um diretório de projeto isolado, com estado estruturado, verificação real por execução e rastreabilidade completa.

## Índice

1. [Visão geral](#1-visão-geral)
2. [Arquitetura](#2-arquitetura)
3. [TaskState](#3-taskstate)
4. [Fluxo de execução](#4-fluxo-de-execução)
5. [Análise e correção de erros](#5-análise-e-correção-de-erros)
6. [Ciclo de correção](#6-ciclo-de-correção)
7. [Planner](#7-planner)
8. [Executor](#8-executor)
9. [Ferramentas](#9-ferramentas)
10. [Providers e LLMs](#10-providers-e-llms)
11. [Execução e sandbox](#11-execução-e-sandbox)
12. [Otimizações](#12-otimizações)
13. [Configuração](#13-configuração)
14. [Testes](#14-testes)
15. [Observabilidade](#15-observabilidade)
16. [Estrutura do projeto](#16-estrutura-do-projeto)
17. [Limitações atuais](#17-limitações-atuais)

## 1. Visão geral

O AIDev recebe um **objetivo** (por exemplo, "crie um gerenciador de notas em Python com testes") e trabalha de forma iterativa: a cada ciclo ele decide a próxima ação, executa ferramentas de leitura/escrita/execução sobre os arquivos do projeto, observa o resultado, roda testes de verdade e corrige falhas até poder declarar o objetivo concluído — ou falhar explicitamente quando a continuação é impossível.

A diferença em relação a simplesmente pedir código a um LLM está no ciclo fechado de verificação:

- **Planejamento explícito**: cada passo é uma decisão registrada (`task`, `finish` ou `fail`), nunca uma ação implícita.
- **Estado estruturado persistente**: requisitos, decisões, progresso, problemas e verificações vivem num `TaskState` versionado em disco, não apenas na memória da conversa.
- **Validação por execução real**: nada é considerado pronto por ter sido escrito; testes, builds e checagens estáticas rodam de verdade (isolados em sandbox) antes de qualquer conclusão.
- **Correção orientada por diagnóstico**: falhas de teste viram problemas estruturados (erro observado, causa provável, solução sugerida, arquivos afetados) que guiam as correções seguintes.
- **Portas de segurança antes de concluir**: checagem estática do projeto, ausência de falhas pendentes e verificação final semântica precisam passar antes de um `finish` ser aceito.

## 2. Arquitetura

```text
Objetivo do usuário
        ↓
Canonicalização / Interpretação  (prompt canônico em inglês + TaskState inicial)
        ↓
TaskState  ──────────────→  persistido em .aidev/task_state.json
        ↓
Planner  ←──  TASK STATE + contexto operacional + ferramentas + regras
   ↓      ↓
 task   finish/fail ──→ portas de verificação ──→ Aceito / Bloqueado
   ↓
Executor  ←──  TASK + TASK STATE + resultados de dependências + KNOWN PROBLEMS
   ↓
Tools  (leitura / análise / escrita / execução)
   ↓
Resultado observado
   ↓
TaskState atualizado  +  traces
   ↓
Teste falhou? ──→ Unified Error Analysis ──→ problemas estruturados
   ↓                                             ↓
Test Gate: retest bloqueado              Correções (Executor)
enquanto houver pendências                        ↓
                                         retest permitido
                                              ↓
                                        verde → resolvido
```

Principais componentes e relações:

- **Runner** (`app/agent/runner.py`): orquestra todo o ciclo — inicializa o estado, chama Planner e Executor, executa ferramentas (com paralelização segura de leituras), atualiza memórias e TaskState, aplica portas de verificação e persiste o estado. Nada de decisão semântica mora aqui; ele é o esqueleto determinístico do loop.
- **Planner** (`app/agent/planning/`): decide a próxima ação de alto nível (`task` com ferramenta/argumentos/dependências, `finish` com resumo do resultado, ou `fail` com motivo).
- **Executor** (`app/agent/execution/`): transforma a decisão do Planner em argumentos finais de execução (por exemplo, gera o conteúdo completo de um arquivo a partir de uma breve descrição) e valida a chamada.
- **TaskState** (`app/agent/taskstate/`): fonte estruturada de verdade sobre a tarefa — ver seção 3.
- **Memórias de contexto** (`app/agent/context/`): resumo persistente do projeto, histórico determinístico de ações, checklists de objetivo e de erros, verificação final.
- **Análise de erros** (`app/agent/errors/`): transforma saídas de teste/build com falha em problemas estruturados — ver seção 5.
- **Tools** (`app/tools/`): operações reais sobre o projeto — ver seção 9.
- **LLM** (`app/llm/`): clientes, providers e roteamento com fallback — ver seção 10.

## 3. TaskState

O `TaskState` é a representação estruturada e persistente do que o agente sabe sobre a tarefa. Ele existe porque resumos narrativos reescritos a cada iteração perdem detalhes, e espalhar o mesmo fato em meia dúzia de textos diferentes gera divergência. Com o estado estruturado, cada fato tem um lugar canônico, uma origem declarada e um ciclo de vida explícito.

Informações representadas:

| Campo | Conteúdo |
|---|---|
| `original_prompt` / `canonical_prompt` | Texto exato do usuário e versão canônica em inglês; o original nunca é descartado |
| `objective` | Objetivo de trabalho derivado do prompt canônico |
| `requirements` | Requisitos explícitos extraídos do prompt |
| `constraints` | Restrições explícitas (limites, proibições, condições) |
| `ambiguities` | Ambiguidades detectáveis no pedido |
| `plan` | Itens de plano espelhados do checklist, com status (`pending` → `in_progress` → `completed`, ou `blocked`) |
| `decisions` | Cada decisão validada do Planner (ação, ferramenta, arquivo, iteração) |
| `progress` | Uma entrada compacta por execução (ferramenta, sucesso/falha, arquivo) |
| `problems` | Problemas estruturados de execução/teste, com status e vínculo com correções |
| `corrections` | Correções aplicadas pelo Executor, vinculadas aos problemas que tentam resolver |
| `verification` | Vereditos das portas de verificação (qual porta, passou/bloqueou, motivo) |

Cada entrada carrega sua **proveniência**: texto do usuário, texto canônico, estrutura interpretada, fato observado em execução ou fato verificado por teste. Nada é marcado como verificado por decisão — só por execução real.

O estado é atualizado por métodos centralizados chamados pelo Runner (nunca por mutação espalhada), serializado de forma determinística (`to_dict`/`from_dict`), exposto de forma compacta e legível (`render_compact`) para consumo pelos prompts, fotografado nos traces e persistido por projeto em `projects/<nome>/.aidev/task_state.json` (escrita atômica; ao iniciar, um estado anterior só é reaproveitado se for da mesma tarefa, identificada por hash do prompt — caso contrário é arquivado e um novo começa).

## 4. Fluxo de execução

1. **Recebimento do objetivo**: via `--objective`, `--objective-file` ou o prompt de demonstração padrão. O texto original é preservado integralmente.
2. **Canonicalização/interpretação**: o prompt é normalizado e, quando necessário, traduzido para um prompt canônico em inglês (tradução apenas linguística — sem planejar nem inventar requisitos); um interpretador determinístico extrai objetivo, requisitos, restrições e ambiguidades para o `TaskState` inicial.
3. **Planejamento**: o Planner recebe objetivo canônico, bloco `TASK STATE`, contexto operacional (resumo, checklists, histórico, arquivos), erros e ferramentas, e retorna `task`, `finish` ou `fail` em JSON.
4. **Execução**: o Executor resolve os argumentos finais (por exemplo, o conteúdo completo do arquivo), valida contra o schema da ferramenta e o Runner a executa.
5. **Uso das ferramentas**: leituras e análises independentes podem rodar em paralelo; escritas e comandos sempre sequenciais. Dependências de uma tarefa (ex.: ler um arquivo antes de editá-lo) executam antes da ação principal.
6. **Observação dos resultados**: todo resultado alimenta o histórico determinístico, o resumo do projeto (quando a operação muda o estado), o TaskState e os traces.
7. **Testes**: comandos de teste/build e a checagem estática do projeto executam de verdade, isolados (ver seção 11); cada execução gera veredito estruturado (status, exit code, stdout/stderr).
8. **Tratamento de falhas**: saída de teste com falha passa por análise unificada (seção 5), vira problema estruturado e bloqueia conclusões até ser tratada.
9. **Correção**: o Executor recebe o resumo dos problemas conhecidos e corrige com base em diagnóstico verificado em leitura, não em palpite.
10. **Verificação final**: `finish` só é aceito após checagem estática íntegra, zero falhas pendentes e verificação semântica final aprovada. Qualquer bloqueio devolve o agente ao ciclo com o motivo registrado.

## 5. Análise e correção de erros

Quando um teste ou build falha, o `UnifiedErrorAnalyzer` (`app/agent/errors/`) processa o resultado **uma única vez** e produz um `UnifiedErrorAnalysis` que alimenta, a partir da mesma análise, o TaskState, o checklist de erros e os contextos do Planner e do Executor. Antes, checklist e analisador faziam duas chamadas LLM independentes sobre o mesmo output; agora há uma só (com fallback 100% determinístico que nunca inventa nada em caso de timeout, resposta inválida ou erro de provider).

A análise combina duas camadas:

- **Fatos determinísticos** (`FailureFact`, extraídos por código — nunca pedidos ao LLM): identificador do teste, linha de erro observada, arquivos mencionados no output, exit code, comando. O determinístico também delimita o que o LLM pode afirmar: arquivos afetados só entram se aparecem de fato na saída.
- **Hipóteses de trabalho** (única chamada LLM): para cada falha, erro observado, **causa provável** e **solução sugerida** — sempre marcadas como hipótese, nunca como certeza. Sem evidência suficiente, o resultado é `probable_cause = unknown` e `suggested_solution = investigate`, o que é preferível a inventar.

Cada problema estruturado carrega: identificador estável, teste de origem, erro observado, causa provável, solução sugerida, arquivos afetados e status (`pending` → `in_progress` → `pending_verification` → `resolved`; `blocked` quando uma tentativa falha; `invalidated` quando nova evidência derruba o diagnóstico). Correções registram quais arquivos mudaram e a quais problemas se vinculam (por sobreposição de arquivos, ou a todos os abertos quando os arquivos são desconhecidos).

O **checklist de erros** continua existindo como projeção operacional da mesma análise (uma linha por falha, com arquivos), e é ele que bloqueia conclusões prematuras. Uma única análise pode — e deve — produzir **múltiplos problemas** para o mesmo resultado (ex.: 4 testes falhando geram 4 problemas rastreáveis, não um texto único).

## 6. Ciclo de correção

```text
Teste falha
    ↓
Análise unificada do resultado (1 chamada)
    ↓
Problemas estruturados (pending)
    ↓
Investigação (leituras; Executor confirma a hipótese)
    ↓
Correção (write_file / comando que altera o projeto)
    ↓
Problemas vinculados → pending_verification
    ↓
Test Gate permite retest (sem pendências sem correção)
    ↓
Verde → resolved  |  Nova falha → novo ciclo
```

**Test Gate**: antes de executar qualquer comando classificado como teste, build ou verificação (incluindo `check_project` como tarefa), o Runner verifica se há problemas `pending`, `in_progress` ou `blocked`. Havendo, a execução é bloqueada com a mensagem estruturada `TEST_BLOCKED_BY_PENDING_PROBLEMS` — um estado esperado, não um erro de infraestrutura: não contamina o checklist, não conta como falha real e orienta o Planner a corrigir em vez de retestar. Regras implementadas:

- O bloqueio vale para comandos de teste/verificação; comandos de investigação (`ls`, `cat`, `grep`, `--version`, `git diff` etc.) e comandos necessários à correção continuam permitidos.
- Correções aplicadas liberam a verificação: basta que todos os problemas tenham recebido correção; a confirmação vem do teste, nunca de autodeclaração.
- Nova falha após retest gera **novo** problema (deduplicação conservadora por comando + teste + erro normalizado; mesmo teste com erro diferente invalida o diagnóstico antigo em vez de duplicar).
- Sem correção nova, repetir o mesmo teste continua bloqueado — é assim que o agente evita o ciclo testa→falha→retesta sem progresso.

## 7. Planner

O Planner decide, a cada iteração, a próxima ação (`task` com ferramenta, argumentos e dependências de leitura; `finish` com descrição do resultado; `fail` com motivo). Ele recebe: objetivo canônico, bloco `TASK STATE` (estado semântico compacto, incluindo problemas com arquivos/causa/correção sugerida), contexto operacional (resumo do projeto, checklist do objetivo, checklist de erros atuais, erros proibidos, histórico determinístico de ações, comandos conhecidos, lista real de arquivos), resultados recentes e os schemas das ferramentas — além das regras normativas (só JSON válido, dependências só com ferramentas de análise, validação real antes de `finish`, etc.).

Mecanismos relacionados:

- **Memória operacional**: histórico real das ações da run (ferramenta, argumentos resumidos, sucesso/falha) mais a lista de arquivos consultada no disco a cada iteração — fonte determinística que corrige eventuais distorções do resumo narrativo.
- **Compactação de contexto**: schemas de ferramentas em forma mínima, janela de histórico recente com preservação de falhas, resultados preservando veredito (status/exit code no início, cauda relevante), teto na cauda de blocos de erro e resumo limitado — tudo estrutural, nunca truncamento cego; desligável por configuração.
- **Short Repair**: se a decisão vem inválida (JSON quebrado, ferramenta inexistente, schema errado), o Planner recebe um prompt curto só com a decisão anterior, o erro e uma dica direcionada — sem reenviar o contexto inteiro. Se o reparo curto falha, há fallback para retry com contexto completo, dentro do mesmo orçamento de 5 tentativas (`MAX_PLANNER_ATTEMPTS`).
- **Retries e respostas inválidas**: até 5 tentativas por decisão; respostas vazias, tool calls diretas ou ações desconhecidas geram erro tratado; erros repetidos recebem correção reforçada e memória permanente de "erros proibidos" na run; estagnação (iterações sem alteração real) e repetição de tarefas disparam avisos explícitos no contexto.

## 8. Executor

O Executor (`TaskDecisionMaker`) transforma a decisão do Planner em ação concreta: recebe a task (ferramenta indicada + argumentos preliminares, onde conteúdo de arquivo vem como breve descrição), o contexto das dependências já executadas e um bloco `KNOWN PROBLEMS`, e retorna os argumentos finais válidos (para `write_file`, o conteúdo completo do arquivo). Ele não cria tarefas nem dependências novas, não troca de ferramenta e não altera a finalidade da task — qualquer violação gera erro e retry (até 5 tentativas, `MAX_EXECUTOR_ATTEMPTS`), com validação de schema antes da execução real.

- **Contexto recebido**: objetivo, task decidida, resultados das dependências (limitados por tamanho) e `KNOWN PROBLEMS` — resumo compacto e limitado dos problemas em aberto (erro, causa provável, solução sugerida, arquivos, status), apresentado explicitamente como hipótese a verificar com leitura antes de corrigir.
- **Relação com TaskState**: cada execução principal vira uma entrada de progresso; execuções que alteram arquivos com problemas em aberto geram `Correction` vinculada e movem esses problemas para verificação pendente.
- **Relação com ferramentas**: o Executor nunca chama ferramentas diretamente (retorna só JSON); é o Runner quem executa via registro central, com validação de path traversal e registro de métricas por ferramenta.

## 9. Ferramentas

Sete ferramentas, em três grupos:

**Leitura (puras, paralelizáveis entre si):**

| Ferramenta | Finalidade |
|---|---|
| `list_files` | Lista arquivos do projeto |
| `read_file` | Lê o conteúdo de um arquivo |
| `find_references` | Localiza onde um símbolo é usado no projeto |
| `list_symbols` | Lista símbolos exportados/definidos por um arquivo (funções, classes etc.) |

**Análise:**

| Ferramenta | Finalidade |
|---|---|
| `check_project` | Checagem estática de todos os arquivos (sintaxe/compilação conforme a linguagem detectada); usada também como porta automática antes de aceitar `finish` |
| `run_command` | Executa comandos shell (testes, builds, programas) — ver seção 11. Apesar de classificada como análise (pode ser dependência para coletar evidência), nunca é paralelizada: qualquer comando pode alterar arquivos |

**Escrita:**

| Ferramenta | Finalidade |
|---|---|
| `write_file` | Cria ou sobrescreve arquivos; única ferramenta classificada como execução |

Ferramentas de leitura/investigação não podem ser tarefa principal isolada (devem vir como dependências da ação real), exceto logo após falha de teste/build ou em objetivos puramente de análise — regra validada automaticamente, com orçamento periódico de investigação.

## 10. Providers e LLMs

Todos os providers implementam a mesma interface (`LLMProvider`) sobre uma base compatível com a API da OpenAI; disponíveis: **Groq**, **OpenRouter** e **Agnes** (`main.py`). O `LLMRouter` percorre os providers configurados e, diante de rate limit, falha de conexão ou erro transitório de API, avança para o próximo, com espera e novas rodadas limitadas antes de desistir. Chamadas registram componente (`Planner`, `TaskDecisionMaker`, `ProjectSummaryUpdater`, `ErrorAnalyzer`, etc.), iteração, tamanhos de prompt/resposta, latência e uso de tokens.

Uso rastreado (`UsageTracker`): totais e por provider, detalhamento por componente (chamadas, tokens de prompt/completion, latência média/total/máxima, retries, falhas), estatísticas do atualizador de resumo e decomposição do contexto do Planner por seção (estimado vs. real do provider, por iteração).

## 11. Execução e sandbox

`run_command` executa comandos shell dentro da pasta do projeto com `stdin` simulável e timeout configurável, sempre reportando `STATUS` (sucesso/falha), exit code, stdout e stderr — formato que todo o resto do sistema consome de forma estruturada.

- **Modo Docker** (`AIDEV_SANDBOX_MODE=docker`, padrão): cada comando roda num container descartável com limites de memória/CPU/PIDs. Sem acesso à rede por padrão (`AIDEV_SANDBOX_NETWORK_MODE=none`); o modo `restricted` libera apenas domínios autorizados via proxy Squid com allowlist (npm, PyPI, Go modules, crates.io, Maven, RubyGems, Packagist, GitHub), com rede e proxy verificados/criados automaticamente na inicialização.
- **Outro valor**: execução direta no sistema, sem isolamento (útil para desenvolvimento/testes locais).

Time-outs matam o grupo de processos inteiro (sem órfãos), e comandos que não terminam sozinhos (servidores) são desencorajados nas instruções do Planner em favor de comandos com fim (testes, builds).

## 12. Otimizações

Cada mecanismo resolve um desperdício ou risco específico; todos são reversíveis por configuração:

- **Short Repair**: retries do Planner sem reenviar ~todo o contexto a cada tentativa — só decisão anterior + erro + dica.
- **Compactação de contexto**: Planner, Executor e resultados trafegam representações mínimas e estruturadas (schemas compactos, janela de histórico, vereditos preservados, tetos por seção) em vez de dumps integrais.
- **Smart Summary**: o resumo narrativo não é regenerado após operações que não mudam o estado (leituras, análises, inspeções, testes bloqueados) — a evidência já está no contexto e no histórico.
- **Paralelização segura**: dependências puramente de leitura executam em lote concorrente com resultados recompostos em ordem determinística; qualquer operação fora do conjunto puro segue sequencial.
- **Classificação de pureza**: cada ferramenta declara `pure` (padrão conservador `False`); `run_command` é sempre não-puro mesmo para comandos com cara de leitura — sem sniffing de shell.
- **Redução de chamadas**: resultados de dependência limitados, listagem de arquivos reenviada só quando muda, checagem de `finish` com cache entre tentativas consecutivas.
- **Análise unificada de erros**: uma chamada por resultado com falha alimenta TaskState, checklist, Planner e Executor (antes: duas análises independentes do mesmo output).
- **Traces e logs com rotação**: traces JSONL por run com poda dos mais antigos (padrão: últimos 20 arquivos / 50 MB totais); log da aplicação com rotação por tamanho (padrão: 10 MB, 3 backups).
- **Métricas de uso e performance**: por componente e por ferramenta (chamadas, tokens, latência, falhas, comandos), com resumo de performance ao fim de cada run.

## 13. Configuração

Variáveis principais (nomes exatos; flags `AIDEV_*` ligadas com `1`, desligadas com `0`):

| Variável | Efeito (padrão) |
|---|---|
| `GROQ_API_KEY` / `GROQ_MODEL`, `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`, `AGNES_API_KEY` / `AGNES_MODEL` | Credenciais dos providers (só entram em uso os completos) |
| `AIDEV_PROJECTS_DIR` | Raiz dos projetos (`projects`) |
| `AIDEV_MAX_ITERATIONS` | Teto de iterações por run (`50`) |
| `AIDEV_EXECUTION_TIMEOUT_SECONDS` | Timeout de comandos (`15`) |
| `AIDEV_LLM_MAX_OUTPUT_TOKENS` | Teto de tokens de saída (`8192`) |
| `AIDEV_RATE_LIMIT_MAX_WAIT_ROUNDS` / `AIDEV_RATE_LIMIT_BASE_WAIT_SECONDS` / `AIDEV_RATE_LIMIT_MAX_WAIT_SECONDS` | Backoff entre providers (`3` / `10` / `90`) |
| `AIDEV_PARALLEL_TOOLS` | Lote paralelo de leituras (`1`) |
| `AIDEV_SMART_SUMMARY` / `AIDEV_COMPACT_CONTEXT` / `AIDEV_COMPACT_EXECUTOR` / `AIDEV_COMPACT_PLANNER` | Otimizações de chamadas/contexto (`1`) |
| `AIDEV_TASK_STATE_CONTEXT` / `AIDEV_TASK_STATE_PERSIST` / `AIDEV_CANONICAL_OBJECTIVE` / `AIDEV_PT_ASCII_TRANSLATION` / `AIDEV_TASK_PLAN_SYNC` | Integração do TaskState (`1`) |
| `AIDEV_ERROR_ANALYZER` / `AIDEV_ERROR_TEST_GATE` | Análise unificada e gate de retest (`1`) |
| `AIDEV_TRACE_DIR` / `AIDEV_TRACE_KEEP_FILES` / `AIDEV_TRACE_MAX_TOTAL_MB` | Traces (padrão: temp do sistema + `20` arquivos / `50` MB) |
| `AIDEV_LOG_LEVEL` / `AIDEV_LOG_FILE` / `AIDEV_LOG_MAX_MB` / `AIDEV_LOG_BACKUPS` | Log (`INFO` / `aidev.log` / `10` MB / `3`) |
| `AIDEV_SANDBOX_MODE` | `docker` (isolado) ou outro valor (execução direta) |
| `AIDEV_SANDBOX_DOCKER_IMAGE` / `AIDEV_SANDBOX_MEMORY_LIMIT` / `AIDEV_SANDBOX_CPU_LIMIT` / `AIDEV_SANDBOX_PIDS_LIMIT` | Imagem e limites do container |
| `AIDEV_SANDBOX_NETWORK_MODE` (`none`/`restricted`) / `AIDEV_SANDBOX_NETWORK_NAME` / `AIDEV_SANDBOX_PROXY_*` | Rede do sandbox e proxy de egresso |

Uso básico:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # e preencha ao menos um provider

python main.py --project meu_projeto --objective "Descreva aqui o que construir."
python main.py --project meu_projeto --objective-file objetivo.txt
python main.py          # demonstração: gerenciador de hábitos em terminal
python main.py --help   # todas as opções
```

## 14. Testes

Requer Python 3.12+ com o ambiente acima instalado. A suíte atual contém **618 testes** cobrindo parsers de decisão, ferramentas e proteções de path traversal, validadores, memórias, Planner/Executor, retries e Short Repair, TaskState e canonicalização, análise de erros e Test Gate, checklist, Finish Gate e verificação final, roteamento/fallback de providers, paralelização, sandbox, traces, métricas e classificações de pureza.

```bash
pytest -q                                  # suíte completa
pytest tests/test_runner.py -q             # um arquivo
pytest tests/test_runner.py::test_x -q     # um teste específico
```

## 15. Observabilidade

- **Traces** (`ExecutionTrace`, um JSONL por run): início/fim de run e iterações, decisões do Planner, retries (curto vs. completo, com tamanhos), erros do Executor, resultados de ferramentas e de testes, batches paralelos, atualizações e skips de resumo, análises de erro (problemas criados/invalidados, fallback), bloqueios de teste e de finish, gates de verificação, snapshot inicial do TaskState e persistências. Falhas de escrita nunca quebram a run.
- **Logs**: log interno em arquivo com rotação (`aidev.log`) mais console; níveis via `AIDEV_LOG_LEVEL`.
- **Métricas**: `UsageTracker` (tokens por componente/provider, latências, tamanhos de prompt, decomposição do contexto do Planner) e `AgentStats`/tool tracker (iterações, retries, testes executados/passados, batches paralelos e tempo economizado, tempo por ferramenta, skips de resumo, bloqueios) com resumo de performance impresso ao fim de cada execução.
- **Eventos de CLI**: progresso legível no terminal a cada transição importante (início/fim de Planner/Executor/tools, erros, verificação final, conclusão).

## 16. Estrutura do projeto

```text
main.py                  CLI e montagem do agente
app/config.py            configuração central (env)
app/logging_config.py    logging com rotação
app/agent/runner.py      orquestração do ciclo
app/agent/planning/      Planner (decisão task/finish/fail, Short Repair)
app/agent/execution/     Executor (argumentos finais, validação)
app/agent/taskstate/     TaskState, canonicalização, persistência
app/agent/errors/        análise unificada de erros
app/agent/context/       resumo, memórias, checklists, verificação final
app/agent/parallel.py    batches paralelos seguros (+ pureza)
app/agent/trace.py       traces JSONL com rotação
app/agent/perf.py        estatísticas de performance
app/llm/                 cliente, router/fallback, providers, usage
app/tools/               filesystem/, analysis/, execution/
app/cli/                 argumentos e eventos de terminal
projects/<nome>/         projetos gerados (+ .aidev/ com resumo e TaskState)
tests/                   suíte de testes
docker/                  imagem do sandbox e proxy de egresso
```

## 17. Limitações atuais

- Objetivos muito complexos podem exceder `AIDEV_MAX_ITERATIONS` sem concluir; o limite evita loops infinitos, não garante conclusão.
- Sem rede no sandbox por padrão, comandos que precisam baixar dependências ou ferramentas ausentes da imagem falham com exit code (reportado normalmente, não como exceção).
- A classificação de comandos de teste/build é heurística (palavras-chave); comandos ambíguos podem ser tratados de forma conservadora (ex.: bloqueados pelo Test Gate ou incluídos no checklist).
- A tradução do prompt para o inglês canônico é linguística e pode ser desativada; diagnósticos do analisador (`probable_cause`, `suggested_solution`) são hipóteses a confirmar por leitura, nunca certezas.
- Falhas de infraestrutura (timeout do sandbox, erro de provider esgotado) não contam como evidência de teste: preservam o estado anterior em vez de liberar conclusões.
- Provedores e modelos dependem de configuração externa válida; sem nenhuma credencial completa o programa informa o que falta antes de qualquer chamada de rede.
