import pytest

from app.llm.models import Usage
from app.llm.usage import UsageTracker


def test_record_null_usage_is_noop():
    tracker = UsageTracker()
    tracker.record(None, "groq")
    assert tracker.calls == 0
    assert tracker.total.total_tokens == 0


def test_record_accumulates_totals():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq")
    tracker.record(Usage(200, 20, 220), "groq")

    assert tracker.calls == 2
    assert tracker.total.prompt_tokens == 300
    assert tracker.total.completion_tokens == 30
    assert tracker.total.total_tokens == 330


def test_record_tracks_by_provider():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq")
    tracker.record(Usage(200, 20, 220), "openrouter")

    assert tracker.by_provider["groq"].total_tokens == 110
    assert tracker.by_provider["openrouter"].total_tokens == 220


def test_record_unknown_provider_when_none():
    tracker = UsageTracker()
    tracker.record(Usage(50, 5, 55), None)

    assert "unknown" in tracker.by_provider
    assert tracker.by_provider["unknown"].total_tokens == 55


def test_record_tracks_by_component():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq", component="Planner")
    tracker.record(Usage(50, 5, 55), "groq", component="Planner")
    tracker.record(Usage(80, 8, 88), "groq", component="TaskDecisionMaker")

    assert tracker._by_component["Planner"].prompt_tokens == 150
    assert tracker._by_component["Planner"].completion_tokens == 15
    assert tracker._by_component["TaskDecisionMaker"].prompt_tokens == 80
    assert tracker._by_component["TaskDecisionMaker"].completion_tokens == 8
    assert tracker._component_calls["Planner"] == 2
    assert tracker._component_calls["TaskDecisionMaker"] == 1


def test_record_without_component_does_not_pollute_breakdown():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq")
    tracker.record(Usage(50, 5, 55), "groq", component="Planner")

    assert tracker.total.prompt_tokens == 150
    assert "Planner" in tracker._by_component
    assert "unknown" not in tracker._by_component


def test_record_stores_iteration():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq", component="Planner", iteration=5)

    records = tracker.get_records()
    assert len(records) == 1
    assert records[0].iteration == 5
    assert records[0].component == "Planner"


def test_record_stores_prompt_and_completion_chars():
    tracker = UsageTracker()
    tracker.record(
        Usage(100, 10, 110),
        "groq",
        component="Planner",
        prompt_chars=400,
        completion_chars=50,
    )

    records = tracker.get_records()
    assert records[0].prompt_chars == 400
    assert records[0].completion_chars == 50
    assert records[0].total_chars == 450


def test_summary_format():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq")
    tracker.record(Usage(200, 20, 220), "openrouter")

    summary = tracker.summary()
    assert "Chamadas à LLM: 2" in summary
    assert "Tokens totais: 330" in summary
    assert "prompt: 300" in summary
    assert "completion: 30" in summary
    assert "groq: 110 tokens" in summary
    assert "openrouter: 220 tokens" in summary


def test_breakdown_format():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq", component="Planner")
    tracker.record(Usage(50, 5, 55), "groq", component="Planner")
    tracker.record(Usage(80, 8, 88), "groq", component="TaskDecisionMaker")

    breakdown = tracker.breakdown()
    assert "TOKEN USAGE BREAKDOWN" in breakdown
    assert "Planner" in breakdown
    assert "TaskDecisionMaker" in breakdown
    assert "253" in breakdown
    assert "|     3 |" in breakdown
    assert "230" in breakdown


def test_breakdown_orders_by_total_tokens_descending():
    tracker = UsageTracker()
    tracker.record(Usage(10, 1, 11), "groq", component="Small")
    tracker.record(Usage(100, 10, 110), "groq", component="Large")
    tracker.record(Usage(50, 5, 55), "groq", component="Medium")

    breakdown = tracker.breakdown()
    lines = [l for l in breakdown.split("\n") if l.strip().startswith("Large") or l.strip().startswith("Medium") or l.strip().startswith("Small")]
    assert lines[0].startswith("Large")
    assert lines[1].startswith("Medium")
    assert lines[2].startswith("Small")


def test_breakdown_empty_when_no_components():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq")

    result = tracker.breakdown()
    assert "Chamadas à LLM" in result


def test_multiple_providers_accumulated_separately():
    tracker = UsageTracker()
    tracker.record(Usage(100, 10, 110), "groq", component="Planner")
    tracker.record(Usage(200, 20, 220), "openrouter", component="TaskDecisionMaker")

    assert tracker.by_provider["groq"].total_tokens == 110
    assert tracker.by_provider["openrouter"].total_tokens == 220
    assert tracker._by_component["Planner"].total_tokens == 110
    assert tracker._by_component["TaskDecisionMaker"].total_tokens == 220
