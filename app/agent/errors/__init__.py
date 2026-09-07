"""Fase 5 — análise e ciclo de correção de erros de execução/teste.

Error Analyzer fornece HIPÓTESES de trabalho (uma chamada LLM por
resultado com falhas, nunca por erro individual); o Executor verifica
e aplica; o teste confirma; o TaskState registra a verdade operacional.
Fallback determinístico garante que o fluxo legado (error checklist)
continue funcionando sem o Analyzer.
"""

from app.agent.errors.error_analyzer import (
    AnalyzedProblem,
    ErrorAnalysis,
    ErrorAnalyzer,
    FailureFact,
    UnifiedErrorAnalysis,
    UnifiedErrorAnalyzer,
    checklist_description,
    extract_failure_facts,
    normalize_error,
    normalize_test_id,
)

__all__ = [
    "AnalyzedProblem",
    "ErrorAnalysis",
    "ErrorAnalyzer",
    "FailureFact",
    "UnifiedErrorAnalysis",
    "UnifiedErrorAnalyzer",
    "checklist_description",
    "extract_failure_facts",
    "normalize_error",
    "normalize_test_id",
]
