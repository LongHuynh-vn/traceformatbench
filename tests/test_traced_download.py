from pathlib import Path
from unittest.mock import Mock

import pytest

from traceformatbench.data.rcaeval import HF_REPO_ID, download_traced_cases

REQUIRED_FILES = ("traces.parquet", "inject_time.txt")


@pytest.fixture
def mock_download(monkeypatch):
    def write_download(*, filename, local_dir, **kwargs):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Downloaded {filename}", encoding="utf-8")
        return str(path)

    download = Mock(side_effect=write_download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        Mock(side_effect=AssertionError("A traces-only download must not download a snapshot")),
    )
    return download


def downloaded_filenames(download):
    return {call.kwargs["filename"] for call in download.call_args_list}


def test_downloads_only_required_files_and_preserves_input_order(tmp_path, mock_download):
    cases = ["re2tt_ts-auth-service_cpu_1", "re2ob_checkoutservice_cpu_1"]

    paths = download_traced_cases(cases, tmp_path)

    assert paths == [tmp_path / case for case in cases]
    assert downloaded_filenames(mock_download) == {
        f"{case}/{name}" for case in cases for name in REQUIRED_FILES
    }
    for call in mock_download.call_args_list:
        assert call.kwargs["repo_id"] == HF_REPO_ID
        assert call.kwargs["repo_type"] == "dataset"
    for path in paths:
        assert {file.name for file in path.iterdir()} == set(REQUIRED_FILES)


def test_skips_cases_with_both_required_files(tmp_path, mock_download):
    case = "re2ob_checkoutservice_cpu_1"
    case_dir = tmp_path / case
    case_dir.mkdir()
    for name in REQUIRED_FILES:
        (case_dir / name).write_text("Existing contents", encoding="utf-8")

    assert download_traced_cases([case], tmp_path) == [case_dir]

    mock_download.assert_not_called()
    for name in REQUIRED_FILES:
        assert (case_dir / name).read_text(encoding="utf-8") == "Existing contents"


def test_download_is_idempotent(tmp_path, mock_download):
    cases = ["re2ob_checkoutservice_cpu_1", "re2tt_ts-auth-service_cpu_1"]
    first_paths = download_traced_cases(cases, str(tmp_path))
    first_contents = {
        path / name: (path / name).read_bytes() for path in first_paths for name in REQUIRED_FILES
    }
    mock_download.reset_mock()

    assert download_traced_cases(cases, str(tmp_path)) == first_paths

    mock_download.assert_not_called()
    assert {path: path.read_bytes() for path in first_contents} == first_contents


@pytest.mark.parametrize("existing_file", REQUIRED_FILES)
def test_partial_case_downloads_only_missing_file(tmp_path, mock_download, existing_file):
    case = "re2ob_checkoutservice_cpu_1"
    case_dir = tmp_path / case
    case_dir.mkdir()
    existing_path = case_dir / existing_file
    existing_path.write_text("Preserve this file", encoding="utf-8")

    assert download_traced_cases([case], tmp_path) == [case_dir]

    missing_file = next(name for name in REQUIRED_FILES if name != existing_file)
    assert downloaded_filenames(mock_download) == {f"{case}/{missing_file}"}
    assert existing_path.read_text(encoding="utf-8") == "Preserve this file"
    assert (case_dir / missing_file).is_file()


def test_empty_case_directory_still_downloads_required_files(tmp_path, mock_download):
    case = "re2ob_checkoutservice_cpu_1"
    case_dir = tmp_path / case
    case_dir.mkdir()

    assert download_traced_cases([case], tmp_path) == [case_dir]

    assert downloaded_filenames(mock_download) == {f"{case}/{name}" for name in REQUIRED_FILES}
    assert all((case_dir / name).is_file() for name in REQUIRED_FILES)


def test_empty_case_list_does_not_download(tmp_path, mock_download):
    assert download_traced_cases([], tmp_path) == []

    mock_download.assert_not_called()


@pytest.mark.parametrize("missing_file", REQUIRED_FILES)
def test_missing_file_after_download_raises_clear_error(tmp_path, mock_download, missing_file):
    case = "re2ob_checkoutservice_cpu_1"

    def incomplete_download(*, filename, local_dir, **kwargs):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name != missing_file:
            path.write_text("Downloaded contents", encoding="utf-8")
        return str(path)

    mock_download.side_effect = incomplete_download

    with pytest.raises(FileNotFoundError) as error:
        download_traced_cases([case], tmp_path)

    assert case in str(error.value)
    assert missing_file in str(error.value)
