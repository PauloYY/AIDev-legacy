import pytest
from app.llm.json_extraction import extract_json_object


def test_clean_json_is_unchanged():
    content = '{"action": "finish", "content": "ok"}'

    assert extract_json_object(content) == content


def test_strips_markdown_code_fence_with_json_hint():
    content = '```json\n{"action": "finish", "content": "ok"}\n```'

    assert extract_json_object(content) == '{"action": "finish", "content": "ok"}'


def test_strips_plain_markdown_code_fence():
    content = '```\n{"action": "finish", "content": "ok"}\n```'

    assert extract_json_object(content) == '{"action": "finish", "content": "ok"}'


def test_extracts_json_surrounded_by_prose():
    content = (
        "Claro! Aqui está minha decisão:\n\n"
        '{"action": "finish", "content": "ok"}'
        "\n\nEspero que ajude!"
    )

    assert extract_json_object(content) == '{"action": "finish", "content": "ok"}'

def test_parse_json_object_repairs_unescaped_quotes():
    from app.llm.json_extraction import parse_json_object

    # Simula o modelo citando um nome de classe sem escapar as aspas —
    # comum em respostas sobre código Java/C++.
    broken = (
        '{"action": "task", "task": {"tool": "write_file", '
        '"arguments": {"content": "corrigir o método da classe '
        '"QueueManager""}}}'
    )

    result = parse_json_object(broken)

    assert result["action"] == "task"
    assert "QueueManager" in result["task"]["arguments"]["content"]


def test_parse_json_object_repairs_trailing_comma():
    from app.llm.json_extraction import parse_json_object

    broken = '{"action": "finish", "content": "ok",}'

    assert parse_json_object(broken) == {"action": "finish", "content": "ok"}


def test_parse_json_object_raises_on_unrecoverable_garbage():
    import json

    from app.llm.json_extraction import parse_json_object

    with pytest.raises(json.JSONDecodeError):
        parse_json_object("isso não é JSON nem de longe, só texto solto")