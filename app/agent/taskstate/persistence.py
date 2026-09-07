"""Fase 4 (integração) — persistência do TaskState em disco.

Arquivo por projeto: `<projects>/<nome>/.aidev/task_state.json`
(mesmo diretório do `summary.md`, padrão já estabelecido).

Regras:
- Escrita ATÔMICA (tmp + os.replace): processo interrompido nunca
  deixa JSON parcial no arquivo final.
- Falha de persistência NUNCA derruba o agente (retorna ok=False).
- Recuperação SOMENTE para a mesma tarefa (task_id = sha1 do prompt
  canônico): tarefa diferente → arquiva o antigo
  (`task_state.prev-<id>.json`, sem apagar) e começa estado novo.
- Usa `TaskState.to_dict/from_dict` (serialização determinística).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from app.tools.config import get_projects_dir

logger = logging.getLogger(__name__)

DIRECTORY = ".aidev"
FILE_NAME = "task_state.json"
TMP_SUFFIX = ".tmp"
ARCHIVE_SUFFIX = ".archived.json"
MAX_STATE_BYTES = 256 * 1024


def state_path_for_project(project_name: str) -> Path:
    """Caminho do task_state.json (com trava path traversal)."""
    projects_dir = get_projects_dir()
    project_path = (projects_dir / project_name).resolve()
    if not project_path.is_relative_to(projects_dir):
        raise PermissionError(
            "Acesso fora do diretório de projetos não permitido."
        )
    return project_path / DIRECTORY / FILE_NAME


def save_task_state(state, project_name: str) -> dict[str, Any]:
    """Persiste o estado atomicamente. Nunca levanta.

    Retorna {"ok", "path", "bytes"}; em falha, ok=False e motivo em
    "error". Estados acima de MAX_STATE_BYTES são recusados (nunca
    truncados em silêncio — o estado em memória segue intacto).
    """
    try:
        data = state.to_dict()
    except Exception as error:
        return {"ok": False, "path": None, "bytes": 0,
                "error": f"serialize: {type(error).__name__}"}
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2,
                             sort_keys=True)
    except Exception as error:
        return {"ok": False, "path": None, "bytes": 0,
                "error": f"encode: {type(error).__name__}"}
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_STATE_BYTES:
        return {"ok": False, "path": None, "bytes": len(encoded),
                "error": f"too_large: {len(encoded)} bytes"}
    try:
        path = state_path_for_project(project_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + TMP_SUFFIX)
        tmp_path.write_bytes(encoded)
        os.replace(tmp_path, path)
        return {"ok": True, "path": str(path), "bytes": len(encoded)}
    except Exception as error:
        logger.warning("TaskState: falha ao persistir (%s): %s",
                       project_name, error)
        return {"ok": False, "path": None, "bytes": 0,
                "error": f"{type(error).__name__}: {error}"}


def load_task_state(project_name: str):
    """Carrega o estado, ou None (ausente/inválido — nunca levanta)."""
    from app.agent.taskstate.task_state import TaskState

    try:
        path = state_path_for_project(project_name)
    except Exception:
        return None
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("TaskState: arquivo ilegível (%s): %s",
                       project_name, error)
        return None
    try:
        state = TaskState.from_dict(data)
    except Exception:
        return None
    if not state.task_id:
        return None
    return state


def archive_task_state(project_name: str, task_id: str) -> bool:
    """Preserva estado de outra tarefa (renomeia, nunca apaga)."""
    try:
        path = state_path_for_project(project_name)
        if not path.exists():
            return False
        safe_id = "".join(
            c for c in str(task_id or "unknown")[:12] if c.isalnum())
        backup = path.with_name(
            f"task_state.prev-{safe_id or 'unknown'}{ARCHIVE_SUFFIX}")
        if backup.exists():
            return True
        os.replace(path, backup)
        return True
    except Exception as error:
        logger.warning("TaskState: falha ao arquivar (%s): %s",
                       project_name, error)
        return False
