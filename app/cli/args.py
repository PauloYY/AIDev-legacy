import argparse
from dataclasses import dataclass
from pathlib import Path

from app.cli.defaults import DEFAULT_PROJECT_NAME, DEFAULT_PROMPT
from app.config import Config


@dataclass
class CLIArgs:
    project_name: str
    objective: str
    log_level: str
    log_file: str | None
    max_iterations: int
    is_default_demo: bool


def parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = argparse.ArgumentParser(
        prog="aidev",
        description=(
            "AIDev — agente autônomo de desenvolvimento de software. "
            "Recebe um objetivo em linguagem natural e o executa "
            "iterativamente usando ferramentas de leitura/escrita de "
            "arquivos, dentro de um diretório de projeto isolado."
        ),
    )

    parser.add_argument(
        "-p",
        "--project",
        dest="project_name",
        default=None,
        help=(
            "Nome do projeto (subpasta dentro do diretório de projetos). "
            f"Padrão de demonstração: '{DEFAULT_PROJECT_NAME}'."
        ),
    )

    objective_group = parser.add_mutually_exclusive_group()

    objective_group.add_argument(
        "-o",
        "--objective",
        dest="objective",
        default=None,
        help="Objetivo em linguagem natural para o agente executar.",
    )

    objective_group.add_argument(
        "-f",
        "--objective-file",
        dest="objective_file",
        default=None,
        type=Path,
        help="Caminho de um arquivo de texto contendo o objetivo.",
    )

    parser.add_argument(
        "--log-level",
        default=Config.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Nível de log interno (padrão: {Config.log_level}).",
    )

    parser.add_argument(
        "--log-file",
        default=Config.log_file,
        help=(
            "Arquivo para onde o log interno é escrito, além do console. "
            "Use 'none' para desativar o arquivo de log. "
            f"(padrão: {Config.log_file})"
        ),
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=Config.max_iterations,
        help=(
            "Número máximo de iterações do agente antes de abortar "
            f"(padrão: {Config.max_iterations})."
        ),
    )

    args = parser.parse_args(argv)

    is_default_demo = args.project_name is None and args.objective is None and args.objective_file is None

    project_name = args.project_name or DEFAULT_PROJECT_NAME

    if args.objective_file:
        objective = args.objective_file.read_text(encoding="utf-8")
    elif args.objective:
        objective = args.objective
    else:
        objective = DEFAULT_PROMPT

    log_file = None if str(args.log_file).lower() == "none" else args.log_file

    return CLIArgs(
        project_name=project_name,
        objective=objective,
        log_level=args.log_level,
        log_file=log_file,
        max_iterations=args.max_iterations,
        is_default_demo=is_default_demo,
    )
