import pytest

from app.tools.filesystem.list_files import list_files
from app.tools.filesystem.read_file import read_file
from app.tools.filesystem.write_file import write_file
from app.tools.filesystem.find_references import find_references


def test_write_and_read_file(projects_root):
    write_file("proj", "src/main.py", "print('hi')")

    content = read_file("proj", "src/main.py")

    assert content == "print('hi')"
    assert (projects_root / "proj" / "src" / "main.py").exists()


def test_write_file_creates_project_dir(projects_root):
    write_file("new_proj", "a.txt", "conteudo")

    assert (projects_root / "new_proj" / "a.txt").read_text() == "conteudo"


def test_write_file_rejects_new_file_breaking_convention(projects_root):
    write_file("proj", "src/rideService.js", "1")
    write_file("proj", "src/driverService.js", "2")

    with pytest.raises(ValueError):
        write_file("proj", "src/DriverRepository.js", "3")


def test_write_file_accepts_new_file_matching_convention(projects_root):
    write_file("proj", "src/rideService.js", "1")
    write_file("proj", "src/driverService.js", "2")

    write_file("proj", "src/paymentService.js", "3")

    assert (projects_root / "proj" / "src" / "paymentService.js").exists()


def test_write_file_allows_editing_existing_file_regardless_of_style(
    projects_root,
):
    write_file("proj", "src/rideService.js", "1")
    write_file("proj", "src/driverService.js", "2")

    # Arquivo já existente: convenção não bloqueia edição, só criação.
    write_file("proj", "src/rideService.js", "1 atualizado")

    assert (
        projects_root / "proj" / "src" / "rideService.js"
    ).read_text() == "1 atualizado"


def test_write_file_first_file_in_folder_is_never_rejected(projects_root):
    write_file("proj", "src/repositories/DriverRepository.js", "1")

    assert (
        projects_root / "proj" / "src" / "repositories" / "DriverRepository.js"
    ).exists()


def test_list_files(projects_root):
    write_file("proj", "a.py", "1")
    write_file("proj", "sub/b.py", "2")

    files = sorted(list_files("proj"))

    assert files == sorted(["a.py", "sub/b.py"])


def test_list_files_project_not_found(projects_root):
    with pytest.raises(FileNotFoundError):
        list_files("nao_existe")


def test_read_file_not_found(projects_root):
    write_file("proj", "a.py", "1")

    with pytest.raises(FileNotFoundError):
        read_file("proj", "b.py")


def test_read_file_blocks_path_traversal(projects_root):
    write_file("proj", "a.py", "1")

    with pytest.raises(PermissionError):
        read_file("proj", "../outro_projeto/secret.py")


def test_write_file_blocks_path_traversal(projects_root):
    with pytest.raises(PermissionError):
        write_file("proj", "../../etc/passwd", "malicioso")


def test_list_files_blocks_path_traversal(projects_root):
    with pytest.raises(PermissionError):
        list_files("../")


def test_find_references(projects_root):
    write_file("proj", "a.py", "class Foo:\n    pass\n")
    write_file("proj", "b.py", "from a import Foo\n\nFoo()\n")

    references = find_references("proj", "Foo")

    assert len(references) == 3
    assert any("a.py" in ref for ref in references)
    assert any("b.py" in ref for ref in references)


def test_find_references_ignores_git_directory(projects_root):
    write_file("proj", "a.py", "Foo")
    write_file("proj", ".git/objects/x", "Foo")

    references = find_references("proj", "Foo")

    assert len(references) == 1
    assert "a.py" in references[0]