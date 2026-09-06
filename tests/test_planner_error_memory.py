from app.agent.context.planner_error_memory import PlannerErrorMemory


def test_render_empty_by_default():
    memory = PlannerErrorMemory()

    assert "nenhum registrado" in memory.render()


def test_seen_is_false_before_recording():
    memory = PlannerErrorMemory()

    assert memory.seen("erro X") is False


def test_record_makes_seen_true():
    memory = PlannerErrorMemory()

    memory.record("erro X")

    assert memory.seen("erro X") is True


def test_render_lists_recorded_errors():
    memory = PlannerErrorMemory()

    memory.record("erro X")
    memory.record("erro Y")

    rendered = memory.render()

    assert "erro X" in rendered
    assert "erro Y" in rendered
    assert "PROIBIDOS" in rendered


def test_record_does_not_duplicate():
    memory = PlannerErrorMemory()

    memory.record("erro X")
    memory.record("erro X")

    rendered = memory.render()

    assert rendered.count("erro X") == 1


def test_reset_clears_memory():
    memory = PlannerErrorMemory()

    memory.record("erro X")
    memory.reset()

    assert memory.seen("erro X") is False
    assert "nenhum registrado" in memory.render()


def test_render_truncates_to_most_recent():
    memory = PlannerErrorMemory()

    for i in range(memory.MAX_ERRORS_RENDERED + 5):
        memory.record(f"erro {i}")

    rendered = memory.render()

    assert "erro 0" not in rendered
    assert f"erro {memory.MAX_ERRORS_RENDERED + 4}" in rendered
