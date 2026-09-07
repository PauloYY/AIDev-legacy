"""P6 — rotação de logs/traces (determinístico, sem tempo real).

Traces: abaixo do limite não poda; acima poda os mais antigos;
backups preservados até o limite e removidos além dele; log atual
nunca removido; JSONL segue válido; defaults/env funcionam.
App log: RotatingFileHandler com mesmo formato e defaults seguros.
"""

import json
import logging
import os

import pytest

from app.agent.trace import (
    DEFAULT_TRACE_KEEP_FILES,
    DEFAULT_TRACE_MAX_TOTAL_MB,
    ExecutionTrace,
    prune_old_traces,
    trace_rotation_limits,
)
from app.logging_config import (
    DEFAULT_LOG_BACKUPS,
    DEFAULT_LOG_MAX_MB,
    log_rotation_limits,
    setup_logging,
)


def _make_trace_file(directory, name, size, mtime):
    path = directory / name
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def _trace_names(directory):
    return sorted(p.name for p in directory.iterdir()
                  if p.name.startswith("aidev-trace-"))


# abaixo do limite não rotaciona -------------------------------------------
def test_below_limit_does_not_rotate(tmp_path):
    for i in range(3):
        _make_trace_file(tmp_path, f"aidev-trace-20240101T00000{i}Z-r{i}.jsonl",
                         100, 1000 + i)
    result = prune_old_traces(tmp_path, keep_files=5,
                              max_total_bytes=10_000)
    assert result == {"kept": 3, "removed": 0, "freed_bytes": 0}
    assert len(_trace_names(tmp_path)) == 3


# acima do limite (quantidade) rotaciona os mais antigos --------------------
def test_above_count_limit_rotates_oldest(tmp_path):
    for i in range(5):
        _make_trace_file(tmp_path, f"aidev-trace-20240101T00000{i}Z-r{i}.jsonl",
                         100, 1000 + i)
    result = prune_old_traces(tmp_path, keep_files=2,
                              max_total_bytes=0)
    assert result["removed"] == 3
    assert result["kept"] == 2
    assert result["freed_bytes"] == 300
    remaining = _trace_names(tmp_path)
    assert remaining == [f"aidev-trace-20240101T00000{i}Z-r{i}.jsonl"
                         for i in (3, 4)]


# acima do limite (tamanho) rotaciona ----------------------------------------
def test_above_size_limit_rotates_oldest(tmp_path):
    _make_trace_file(tmp_path, "aidev-trace-20240101T000000Z-old.jsonl",
                     800, 1000)
    _make_trace_file(tmp_path, "aidev-trace-20240101T000001Z-new.jsonl",
                     800, 2000)
    result = prune_old_traces(tmp_path, keep_files=0,
                              max_total_bytes=1000)
    assert result["removed"] == 1
    assert result["kept"] == 1
    assert _trace_names(tmp_path) == [
        "aidev-trace-20240101T000001Z-new.jsonl"]


# arquivos além do limite são removidos; outros arquivos intactos ------------
def test_non_trace_files_untouched(tmp_path):
    keeper = tmp_path / "important.txt"
    keeper.write_text("do not touch")
    _make_trace_file(tmp_path, "aidev-trace-20240101T000000Z-a.jsonl",
                     100, 1000)
    _make_trace_file(tmp_path, "aidev-trace-20240101T000001Z-b.jsonl",
                     100, 2000)
    prune_old_traces(tmp_path, keep_files=1, max_total_bytes=0)
    assert keeper.read_text() == "do not touch"
    assert _trace_names(tmp_path) == [
        "aidev-trace-20240101T000001Z-b.jsonl"]


# log atual nunca é perdido ----------------------------------------------------
def test_current_log_is_never_removed(tmp_path):
    current = _make_trace_file(
        tmp_path, "aidev-trace-20240101T000009Z-current.jsonl", 5000, 500)
    for i in range(3):
        _make_trace_file(tmp_path, f"aidev-trace-20240101T00000{i}Z-r{i}.jsonl",
                         100, 1000 + i)
    result = prune_old_traces(tmp_path, keep_path=current, keep_files=1,
                              max_total_bytes=100)
    assert current.exists()
    assert result["kept"] >= 1  # atual + (nada mais cabe, mas atual fica)


def test_new_log_keeps_being_written_after_prune(tmp_path):
    for i in range(4):
        _make_trace_file(tmp_path, f"aidev-trace-20240101T00000{i}Z-r{i}.jsonl",
                         100, 1000 + i)
    trace = ExecutionTrace(trace_dir=str(tmp_path), run_id="abc123")
    trace.record("iteration_start", iteration=1)
    trace.record("run_end", outcome="success")
    assert trace.path.exists()
    events = trace.read_events()
    assert [e["event"] for e in events] == ["iteration_start", "run_end"]
    assert all(e["run_id"] == "abc123" for e in events)
    # JSONL válido linha a linha.
    with open(trace.path, encoding="utf-8") as handle:
        for line in handle:
            assert json.loads(line)["event"] in (
                "iteration_start", "run_end")


def test_prune_never_raises(tmp_path):
    assert prune_old_traces(tmp_path / "missing-dir") == {
        "kept": 0, "removed": 0, "freed_bytes": 0}
    assert prune_old_traces(None, keep_files=-1,
                            max_total_bytes=-1)["removed"] == 0


# defaults e env -----------------------------------------------------------------
def test_defaults_and_env(monkeypatch):
    monkeypatch.delenv("AIDEV_TRACE_KEEP_FILES", raising=False)
    monkeypatch.delenv("AIDEV_TRACE_MAX_TOTAL_MB", raising=False)
    assert trace_rotation_limits() == (
        DEFAULT_TRACE_KEEP_FILES, DEFAULT_TRACE_MAX_TOTAL_MB * 1024 * 1024)

    monkeypatch.setenv("AIDEV_TRACE_KEEP_FILES", "7")
    monkeypatch.setenv("AIDEV_TRACE_MAX_TOTAL_MB", "2")
    assert trace_rotation_limits() == (7, 2 * 1024 * 1024)

    monkeypatch.setenv("AIDEV_TRACE_KEEP_FILES", "lixo")
    assert trace_rotation_limits()[0] == DEFAULT_TRACE_KEEP_FILES

    monkeypatch.setenv("AIDEV_TRACE_KEEP_FILES", "0")
    monkeypatch.setenv("AIDEV_TRACE_MAX_TOTAL_MB", "0")
    assert trace_rotation_limits() == (0, 0)


def test_log_rotation_defaults_and_env(monkeypatch):
    monkeypatch.delenv("AIDEV_LOG_MAX_MB", raising=False)
    monkeypatch.delenv("AIDEV_LOG_BACKUPS", raising=False)
    assert log_rotation_limits() == (
        DEFAULT_LOG_MAX_MB * 1024 * 1024, DEFAULT_LOG_BACKUPS)

    monkeypatch.setenv("AIDEV_LOG_MAX_MB", "1")
    monkeypatch.setenv("AIDEV_LOG_BACKUPS", "5")
    assert log_rotation_limits() == (1024 * 1024, 5)

    monkeypatch.setenv("AIDEV_LOG_MAX_MB", "invalido")
    assert log_rotation_limits()[0] == DEFAULT_LOG_MAX_MB * 1024 * 1024


# app log rotaciona por tamanho com o mesmo formato -------------------------------
def test_app_log_rotates_by_size(tmp_path):
    log_file = str(tmp_path / "aidev.log")
    setup_logging(level="INFO", log_file=log_file,
                  max_bytes=300, backup_count=2)
    logger = logging.getLogger("test_p6_rotation")
    for i in range(20):
        logger.warning("mensagem de log numero %d com recheio xxxx", i)
    for handler in logging.getLogger().handlers:
        handler.flush()
    backups = sorted(p.name for p in tmp_path.iterdir()
                     if p.name.startswith("aidev.log."))
    assert backups, "esperava ao menos um backup rotacionado"
    assert len(backups) <= 2
    assert (tmp_path / "aidev.log").exists()
    # Formato preservado: "[LEVEL] name: message".
    content = (tmp_path / "aidev.log").read_text(encoding="utf-8")
    assert "test_p6_rotation" in content


def test_app_log_without_file_still_works():
    # Sem arquivo: só stderr, sem exceção.
    setup_logging(level="INFO", log_file=None)
