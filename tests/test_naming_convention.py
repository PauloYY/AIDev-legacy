import pytest

from app.tools.filesystem.naming_convention import (
    classify_name_style,
    detect_folder_convention,
    check_naming_convention,
)


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("rideService", "camelCase"),
        ("RideService", "PascalCase"),
        ("ride-service", "kebab-case"),
        ("ride_service", "snake_case"),
        ("ride.service", "dot.case"),
        ("utils", None),
        ("Repository", None),
        ("index", None),
        ("a", None),
    ],
)
def test_classify_name_style(stem, expected):
    assert classify_name_style(stem) == expected


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("RideService.test", "PascalCase"),
        ("rideService.spec", "camelCase"),
        ("professional.repository", "dot.case"),
        ("professional.repository.test", "dot.case"),
        ("types.d", None),
    ],
)
def test_classify_name_style_strips_test_markers(stem, expected):
    assert classify_name_style(stem) == expected


def test_detect_folder_convention_with_consistent_siblings(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "driverService.js").touch()

    assert detect_folder_convention(tmp_path, ".js") == "camelCase"


def test_detect_folder_convention_ignores_ambiguous_names(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "utils.js").touch()
    (tmp_path / "index.js").touch()

    assert detect_folder_convention(tmp_path, ".js") == "camelCase"


def test_detect_folder_convention_ignores_different_extension(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "ride-service.md").touch()

    assert detect_folder_convention(tmp_path, ".js") == "camelCase"


def test_detect_folder_convention_no_siblings(tmp_path):
    assert detect_folder_convention(tmp_path, ".js") is None


def test_detect_folder_convention_only_ambiguous_siblings(tmp_path):
    (tmp_path / "utils.js").touch()
    (tmp_path / "index.js").touch()

    assert detect_folder_convention(tmp_path, ".js") is None


def test_detect_folder_convention_already_inconsistent(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "driver-service.js").touch()

    # Pasta já misturada antes dessa mudança existir — não forçamos
    # nada retroativamente.
    assert detect_folder_convention(tmp_path, ".js") is None


def test_detect_folder_convention_nonexistent_directory(tmp_path):
    assert detect_folder_convention(tmp_path / "nope", ".js") is None


def test_check_naming_convention_rejects_mismatch(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "driverService.js").touch()

    error = check_naming_convention(tmp_path, ".js", "DriverRepository")

    assert error is not None
    assert "camelCase" in error
    assert "PascalCase" in error


def test_check_naming_convention_accepts_matching_style(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "driverService.js").touch()

    assert check_naming_convention(tmp_path, ".js", "paymentService") is None


def test_check_naming_convention_accepts_ambiguous_new_name(tmp_path):
    (tmp_path / "rideService.js").touch()
    (tmp_path / "driverService.js").touch()

    assert check_naming_convention(tmp_path, ".js", "index") is None


def test_check_naming_convention_no_convention_established(tmp_path):
    assert check_naming_convention(tmp_path, ".js", "DriverRepository") is None