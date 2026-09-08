"""Tests for the delete_file tool (safe workspace file removal)."""

import pytest

from app.tools.filesystem.delete_file import delete_file
from app.tools.filesystem.read_file import read_file
from app.tools.filesystem.write_file import write_file


def test_delete_existing_file(projects_root):
    write_file("proj", "a.py", "conteudo")

    result = delete_file("proj", "a.py")

    assert "DELETE_SUCCESS" in result
    assert "a.py" in result
    assert not (projects_root / "proj" / "a.py").exists()


def test_delete_missing_file_is_controlled_error(projects_root):
    write_file("proj", "a.py", "1")

    with pytest.raises(FileNotFoundError):
        delete_file("proj", "nao_existe.py")


def test_delete_keeps_sibling_files(projects_root):
    write_file("proj", "a.py", "1")
    write_file("proj", "b.py", "2")

    delete_file("proj", "a.py")

    assert read_file("proj", "b.py") == "2"


def test_delete_protects_git_paths(projects_root):
    write_file("proj", ".git/objects/x", "data")

    with pytest.raises(PermissionError, match="protected"):
        delete_file("proj", ".git/objects/x")

    assert (projects_root / "proj" / ".git" / "objects" / "x").exists()


def test_delete_protects_aidev_state(projects_root):
    write_file("proj", ".aidev/task_state.json", "{}")

    with pytest.raises(PermissionError, match="protected"):
        delete_file("proj", ".aidev/task_state.json")


def test_delete_refuses_directories(projects_root):
    write_file("proj", "sub/a.py", "1")

    with pytest.raises(IsADirectoryError):
        delete_file("proj", "sub")


def test_delete_refuses_project_root(projects_root):
    write_file("proj", "a.py", "1")

    with pytest.raises(PermissionError):
        delete_file("proj", ".")


def test_delete_rejects_empty_path(projects_root):
    write_file("proj", "a.py", "1")

    with pytest.raises(ValueError):
        delete_file("proj", "  ")


def test_delete_rejects_path_traversal(projects_root):
    with pytest.raises(PermissionError):
        delete_file("proj", "../../etc/passwd")


def test_delete_registry_integration():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.load_defaults()

    tool = registry.get("delete_file")
    assert tool.pure is False
    assert tool.type.name == "EXECUTION"

    names = {d["function"]["name"] for d in registry.definitions}
    assert "delete_file" in names


def test_delete_schema_validation():
    from app.tools.registry import ToolRegistry
    from app.tools.schema_validator import SchemaValidator

    registry = ToolRegistry()
    registry.load_defaults()
    validator = SchemaValidator()

    validator.validate(registry.get("delete_file"), {
        "project_name": "p",
        "file_path": "a.py",
    })

    with pytest.raises(ValueError):
        validator.validate(registry.get("delete_file"), {
            "project_name": "p",
            # missing file_path
        })


def test_delete_trace_records_file_path():
    from app.agent.trace import extract_file_path, sanitize_arguments

    args = {"project_name": "p", "file_path": "sub/a.py"}
    assert extract_file_path(args) == "sub/a.py"

    sanitized = sanitize_arguments("delete_file", args)
    assert sanitized["file_path"] == "sub/a.py"


def test_edit_trace_omits_large_texts():
    from app.agent.trace import sanitize_arguments

    sanitized = sanitize_arguments("edit_file", {
        "project_name": "p",
        "file_path": "a.py",
        "old_text": "x" * 1000,
        "new_text": "y" * 1000,
    })
    assert sanitized["old_text"] == "<omitted: 1000 chars>"
    assert sanitized["new_text"] == "<omitted: 1000 chars>"
    assert sanitized["file_path"] == "a.py"
