"""Tests for the edit_file tool (precise content-based file editing)."""

import pytest

from app.tools.filesystem.edit_file import edit_file
from app.tools.filesystem.read_file import read_file
from app.tools.filesystem.write_file import write_file


def test_edit_single_occurrence(projects_root):
    write_file("proj", "a.py", "x = 1\nprint(x)\n")

    result = edit_file("proj", "a.py", "x = 1", "x = 2")

    assert "EDIT_SUCCESS" in result
    assert "a.py" in result
    assert "1 occurrence" in result
    assert read_file("proj", "a.py") == "x = 2\nprint(x)\n"


def test_edit_file_not_found(projects_root):
    write_file("proj", "a.py", "conteudo")

    with pytest.raises(FileNotFoundError):
        edit_file("proj", "missing.py", "conteudo", "novo")


def test_edit_old_text_not_found(projects_root):
    write_file("proj", "a.py", "x = 1\n")

    with pytest.raises(ValueError, match="not found"):
        edit_file("proj", "a.py", "y = 999", "y = 2")


def test_edit_multiple_occurrences_rejected(projects_root):
    write_file("proj", "a.py", "a = 1\na = 1\n")

    with pytest.raises(ValueError, match="multiple occurrences"):
        edit_file("proj", "a.py", "a = 1", "a = 2")


def test_edit_preserves_file_on_not_found(projects_root):
    original = "x = 1\nprint(x)\n"
    write_file("proj", "a.py", original)

    with pytest.raises(ValueError):
        edit_file("proj", "a.py", "inexistente", "novo")

    assert read_file("proj", "a.py") == original


def test_edit_preserves_file_on_ambiguity(projects_root):
    original = "foo\nfoo\n"
    write_file("proj", "a.py", original)

    with pytest.raises(ValueError):
        edit_file("proj", "a.py", "foo", "bar")

    assert read_file("proj", "a.py") == original


def test_edit_empty_old_text_rejected(projects_root):
    write_file("proj", "a.py", "conteudo")

    with pytest.raises(ValueError):
        edit_file("proj", "a.py", "", "novo")

    assert read_file("proj", "a.py") == "conteudo"


def test_edit_empty_new_text_allowed(projects_root):
    """Removing a snippet (new_text='') is a legitimate edit."""
    write_file("proj", "a.py", "keep\nremove-me\nkeep2\n")

    result = edit_file("proj", "a.py", "remove-me\n", "")

    assert "EDIT_SUCCESS" in result
    assert read_file("proj", "a.py") == "keep\nkeep2\n"


def test_edit_special_characters(projects_root):
    content = "def f():\n    return 'ação ç ãõ — €'\n"
    write_file("proj", "a.py", content)

    edit_file("proj", "a.py", "'ação ç ãõ — €'", "'ok ✓'")

    assert read_file("proj", "a.py") == "def f():\n    return 'ok ✓'\n"


def test_edit_multiline_block(projects_root):
    content = "start\nline1\nline2\nend\n"
    write_file("proj", "a.py", content)

    edit_file("proj", "a.py", "line1\nline2\n", "single\n")

    assert read_file("proj", "a.py") == "start\nsingle\nend\n"


def test_edit_larger_file(projects_root):
    lines = [f"linha {i}" for i in range(2000)]
    lines[1500] = "ALVO UNICO AQUI"
    write_file("proj", "big.txt", "\n".join(lines) + "\n")

    result = edit_file("proj", "big.txt", "ALVO UNICO AQUI", "SUBSTITUIDO")

    assert "EDIT_SUCCESS" in result
    updated = read_file("proj", "big.txt")
    assert "SUBSTITUIDO" in updated
    assert "ALVO UNICO AQUI" not in updated
    assert updated.count("SUBSTITUIDO") == 1


def test_edit_rejects_path_traversal(projects_root):
    write_file("proj", "a.py", "1")

    with pytest.raises(PermissionError):
        edit_file("proj", "../../etc/passwd", "1", "2")


def test_edit_rejects_directory(projects_root, tmp_path):
    (tmp_path / "proj" / "sub").mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        edit_file("proj", "sub", "x", "y")


def test_edit_guides_toward_specific_snippet(projects_root):
    """Ambiguous error must tell the agent to be more specific."""
    write_file("proj", "a.py", "x = 1\nx = 1\n")

    with pytest.raises(ValueError, match="more specific"):
        edit_file("proj", "a.py", "x = 1", "x = 2")


def test_edit_registry_integration():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.load_defaults()

    tool = registry.get("edit_file")
    assert tool.pure is False
    assert tool.type.name == "EXECUTION"

    names = {d["function"]["name"] for d in registry.definitions}
    assert "edit_file" in names


def test_edit_schema_validation():
    from app.tools.registry import ToolRegistry
    from app.tools.schema_validator import SchemaValidator

    registry = ToolRegistry()
    registry.load_defaults()
    validator = SchemaValidator()

    validator.validate(registry.get("edit_file"), {
        "project_name": "p",
        "file_path": "a.py",
        "old_text": "x",
        "new_text": "y",
    })

    with pytest.raises(ValueError):
        validator.validate(registry.get("edit_file"), {
            "project_name": "p",
            "file_path": "a.py",
            "old_text": "x",
        })
