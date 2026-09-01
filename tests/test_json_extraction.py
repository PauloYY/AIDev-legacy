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