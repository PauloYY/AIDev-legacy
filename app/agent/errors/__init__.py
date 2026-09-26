"""Análise e ciclo de correção de erros de execução/teste."""

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
