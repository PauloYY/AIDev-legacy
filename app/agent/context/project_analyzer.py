import json

from app.llm.client import LLMClient
from app.llm.models import Message


class ProjectAnalyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(
        self,
        project_name: str,
        files: list[str],
    ) -> str:
        prompt = self._build_prompt(
            project_name,
            files,
        )

        response = self.llm.generate(
            messages=[
                Message(
                    role="user",
                    content=prompt,
                )
            ],
            component="ProjectAnalyzer",
        )

        return response.content

    def _build_prompt(
        self,
        project_name: str,
        files: list[str],
    ) -> str:
        files_json = json.dumps(
            files,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
Você é o analisador de projetos de um agente de desenvolvimento.

Analise a estrutura do projeto abaixo e produza um resumo inicial
que será utilizado posteriormente por outro agente para planejar
tarefas.

PROJETO:
{project_name}

ARQUIVOS:
{files_json}

REGRAS:
- Retorne somente o conteúdo do resumo.
- Não use JSON.
- Não invente informações sobre os arquivos.
- Baseie-se somente na estrutura fornecida.
- Indique que o conteúdo dos arquivos ainda não foi analisado.
- Seja conciso.
- O resultado será salvo diretamente em .aidev/summary.md.

O resumo deve conter:

# Project Summary

## Estrutura

Liste os arquivos relevantes.

## Entendimento

Descreva somente o que pode ser inferido pela estrutura.

## Observações

Informe que os arquivos ainda precisam ser analisados quando necessário.
"""