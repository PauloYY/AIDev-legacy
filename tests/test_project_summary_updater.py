"""Compactação do prompt do ProjectSummaryUpdater."""

import pytest

from app.agent.context.project_summary import ProjectSummary
from app.agent.context.project_summary_updater import ProjectSummaryUpdater
from app.agent.execution.task import Task
from app.llm.utils import estimate_tokens


class _FakeLLM:
    """Captura o prompt e retorna um resumo fixo."""

    def __init__(self, summary="novo resumo"):
        self.summary = summary
        self.prompts: list[str] = []

    def generate(self, messages, tools=None, component=None, iteration=None):
        from app.llm.models import LLMResponse, Usage

        self.prompts.append(messages[0].content)
        return LLMResponse(
            content=self.summary,
            tool_calls=[],
            usage=Usage(10, 1, 11),
            provider="fake",
        )


@pytest.fixture
def updater(projects_root):
    llm = _FakeLLM()
    return ProjectSummaryUpdater(llm=llm, summary=ProjectSummary()), llm


def _legacy_prompt(objective, current_summary, task, result) -> str:
    """Reconstrução fiel do prompt antigo (args + result integrais).

    O código antigo interpolava {task.arguments} e {result} diretamente;
    esta réplica serve só para medir a redução.
    """
    return f"""
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
- Limite o resumo a no máximo 300 palavras.
- Não liste nomes de arquivos nem a estrutura de pastas — isso já é
  fornecido separadamente como memória operacional determinística.
  Foque em decisões de design, regras de negócio, convenções
  adotadas e o que ainda falta fazer.
"""


def test_write_file_grande_nao_vai_integral_ao_prompt(updater):
    updater_obj, llm = updater
    big = "CONTEUDO-SECRETO-" * 700
    task = Task(
        tool="write_file",
        arguments={
            "project_name": "p",
            "file_path": "app.py",
            "content": big,
        },
    )
    updater_obj.update(
        objective="obj", project_name="p", task=task, result="ok"
    )
    prompt = llm.prompts[0]
    assert "app.py" in prompt
    assert str(len(big)) in prompt
    assert big not in prompt
    assert "CONTEUDO-SECRETO" not in prompt


def test_argumentos_pequenos_passam_normais(updater):
    updater_obj, llm = updater
    task = Task(
        tool="read_file",
        arguments={"project_name": "p", "file_path": "src/app.py"},
    )
    updater_obj.update(
        objective="obj", project_name="p", task=task, result="conteúdo ok"
    )
    prompt = llm.prompts[0]
    assert "src/app.py" in prompt
    assert "conteúdo ok" in prompt


def test_string_grande_em_outro_argumento_e_truncada(updater):
    updater_obj, llm = updater
    huge = "y" * 5000
    task = Task(
        tool="run_command",
        arguments={"project_name": "p", "command": huge},
    )
    updater_obj.update(
        objective="obj", project_name="p", task=task, result="ok"
    )
    prompt = llm.prompts[0]
    assert huge not in prompt
    assert "chars]" in prompt
    assert huge[:200] in prompt


def test_result_grande_e_truncado(updater):
    updater_obj, llm = updater
    big_result = "STATUS: falha\n" + "L" * 12000
    task = Task(tool="list_files", arguments={"project_name": "p"})
    updater_obj.update(
        objective="obj", project_name="p", task=task, result=big_result
    )
    prompt = llm.prompts[0]
    assert big_result not in prompt
    assert "STATUS: falha" in prompt
    assert "result truncated" in prompt
    assert str(len(big_result)) in prompt


def test_result_pequeno_passam_normal(updater):
    updater_obj, llm = updater
    task = Task(tool="list_files", arguments={"project_name": "p"})
    updater_obj.update(
        objective="obj",
        project_name="p",
        task=task,
        result="Arquivo escrito com sucesso: app.py",
    )
    assert "Arquivo escrito com sucesso: app.py" in llm.prompts[0]


def test_argumentos_originais_nao_modificados(updater):
    updater_obj, _ = updater
    args = {
        "project_name": "p",
        "file_path": "app.py",
        "content": "ORIGINAL-" * 2000,
    }
    snapshot = dict(args)
    task = Task(tool="write_file", arguments=args)
    updater_obj.update(
        objective="obj", project_name="p", task=task, result="ok"
    )
    assert task.arguments == snapshot
    assert args == snapshot
    assert len(args["content"]) == len(snapshot["content"])


def test_fluxo_update_continua_funcionando(updater, projects_root):
    updater_obj, _ = updater
    task = Task(
        tool="write_file",
        arguments={"project_name": "p", "file_path": "a.py", "content": "x"},
    )
    out = updater_obj.update(
        objective="obj", project_name="p", task=task, result="ok"
    )
    assert out == "novo resumo"
    assert ProjectSummary().read("p") == "novo resumo"


def test_medicao_antes_depois_3_10_20kb():
    """Medição obrigatória: compara prompt legado vs compactado."""
    updater_obj = ProjectSummaryUpdater(
        llm=_FakeLLM(), summary=ProjectSummary()
    )
    summary = "resumo atual. "
    for size in (3_000, 10_000, 20_000):
        task = Task(
            tool="write_file",
            arguments={
                "project_name": "p",
                "file_path": "app.py",
                "content": "x" * size,
            },
        )
        result = "STATUS: sucesso\n" + "o" * 500
        before = _legacy_prompt("obj", summary, task, result)
        after = updater_obj._build_prompt("obj", summary, task, result)
        red = (1 - len(after) / len(before)) * 100
        print(
            f"\nwrite_file {size//1000}KB: antes={len(before)} "
            f"({estimate_tokens(before)} toks) depois={len(after)} "
            f"({estimate_tokens(after)} toks) redução={red:.1f}%"
        )
        assert len(after) < len(before) // 2
        assert task.arguments["content"] == "x" * size
