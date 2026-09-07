"""Paralelização segura de operações independentes (Etapa 3).

Escopo propositalmente estreito: executar em paralelo SOMENTE
operações puramente observadoras (leituras), que por definição não
alteram disco/estado e não consomem o resultado umas das outras.

Regras de segurança (não relaxar sem revisão):
- `PURE_READ_TOOLS`: read_file, list_files, find_references,
  list_symbols. `run_command` e `check_project` NUNCA entram aqui:
  o primeiro pode mutar arquivos via shell, o segundo dispara
  compilações com efeitos colaterais (caches, /tmp).
- Writes (`write_file`, `run_command` mutante) continuam sempre
  sequenciais — o Runner executa uma única tool principal por
  iteração, e este módulo nunca é usado para elas.
- Resultados sempre na ordem de entrada (determinismo): o chamador
  faz o bookkeeping (memória, eventos, trace) sequencialmente após
  o batch, como se as operações tivessem rodado em ordem.
- Falha em um item NÃO cancela os demais (mesma semântica do loop
  sequencial atual, que captura por item).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger(__name__)


# Tools comprovadamente sem efeitos colaterais relevantes. Qualquer
# tool fora deste conjunto roda pelo caminho sequencial legado.
# P4: este conjunto deve espelhar exatamente as tools padrão
# declaradas com `pure=True` (ver app/tools/base.py e
# `declared_pure_tool_names()` abaixo). `ToolType.ANALYSIS` NÃO é o
# critério — run_command é ANALYSIS mas nunca entra aqui.
PURE_READ_TOOLS = frozenset({
    "read_file",
    "list_files",
    "find_references",
    "list_symbols",
})

MAX_WORKERS = 4


@dataclass
class BatchItemResult:
    """Resultado de UM item do batch (ordem = ordem de entrada)."""

    success: bool
    value: Any = None
    error: Exception | None = None
    duration_ms: float = 0.0


def is_pure_read_tool(name: Any) -> bool:
    """True somente para tools de leitura comprovadamente puras."""
    return isinstance(name, str) and name in PURE_READ_TOOLS


def all_pure_read(names) -> bool:
    """True se há ≥1 nome e TODOS são leitura pura."""
    names = list(names)
    return len(names) > 0 and all(is_pure_read_tool(n) for n in names)


def declared_pure_tool_names(registry) -> set[str]:
    """Nomes das tools registradas com `pure=True` (P4, só observa).

    Usado para verificar coerência entre a declaração de cada tool e
    `PURE_READ_TOOLS`. Não é usado no caminho quente: o gate do batch
    continua sendo o conjunto explícito acima (não relaxar sem revisão).
    """
    names: set[str] = set()
    tools = getattr(registry, "_tools", None) or {}
    for name, tool in tools.items():
        if bool(getattr(tool, "pure", False)):
            names.add(name)
    return names


def run_concurrent(
    calls: list[Callable[[], Any]],
    max_workers: int = MAX_WORKERS,
) -> tuple[list[BatchItemResult], float]:
    """Executa chamadas independentes em paralelo (threads).

    Retorna (resultados_na_ordem, wall_ms_do_batch). Exceção em um
    item vira `BatchItemResult(success=False)` — os demais continuam.
    `calls` deve conter apenas operações puras/independentes; a
    verificação é responsabilidade do chamador (ver `is_pure_read_tool`).
    """

    if not calls:
        return [], 0.0

    workers = max(1, min(max_workers, len(calls)))
    results: list[BatchItemResult | None] = [None] * len(calls)
    batch_start = time.monotonic()

    def _run(index: int, fn: Callable[[], Any]) -> None:
        start = time.monotonic()
        try:
            value = fn()
        except Exception as error:  # noqa: BLE001 — capturado por item
            results[index] = BatchItemResult(
                success=False,
                value=None,
                error=error,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        else:
            results[index] = BatchItemResult(
                success=True,
                value=value,
                error=None,
                duration_ms=(time.monotonic() - start) * 1000,
            )

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="aidev-parallel"
    ) as pool:
        futures = [
            pool.submit(_run, index, fn)
            for index, fn in enumerate(calls)
        ]
        for future in futures:
            # _run nunca levanta (captura tudo), mas join propaga
            # KeyboardInterrupt/SystemExit normalmente.
            future.result()

    wall_ms = (time.monotonic() - batch_start) * 1000
    return [r for r in results if r is not None], wall_ms


def estimated_saved_ms(
    results: list[BatchItemResult], wall_ms: float
) -> float:
    """Economia estimada do batch: soma(durações) − parede.

    É estimativa honesta (threads têm overhead/GIL), nunca negativa.
    """
    total = sum(r.duration_ms for r in results)
    return max(0.0, total - wall_ms)
