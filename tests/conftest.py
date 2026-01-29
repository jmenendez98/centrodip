from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, Any
from urllib.error import ContentTooShortError, HTTPError, URLError
import urllib.request

import pytest

from centrodip.bedtable import BedTable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Remote datasets: keep the URLs, keep "type" as metadata (NOT a URL).
REMOTE_DATASETS: Dict[str, Dict[str, str]] = {
    "chm13_chr1": {
        "bedmethyl": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/CHM13v2.0_ONT_5mC.chr1.H1L.bedgraph",
        "regions": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/chm13v2.0.cenSatv2.1.chr1.H1L.bed",
        "type": "bedgraph",
    },
    "hg002_chrXY": {
        "bedmethyl": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/HG002v1.1_Q100_ONT_5mC.chrXY.test.pileup.bed",
        "regions": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/hg002v1.1.cenSatv2.0.chrXY.H1L.bed",
        "type": "bedmethyl",
    },
}


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": "centrodip-tests/1.0"})
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (URLError, HTTPError, ContentTooShortError) as exc:
        raise RuntimeError(f"{url} :: {exc}") from exc

    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty file from {url}")

    return destination


@pytest.fixture(scope="session")
def remote_dataset_paths(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Dict[str, Any]]:
    """
    Downloads test datasets once per test session.

    Returns:
      dict like:
        {
          "chm13_chr1": {
              "bedmethyl": Path(...),
              "regions": Path(...),
              "type": "bedgraph",
          },
          ...
        }
    """
    base_dir = tmp_path_factory.mktemp("centrodip_remote_data")
    datasets: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, Exception] = {}

    for dataset_key, spec in REMOTE_DATASETS.items():
        dataset_dir = base_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset_paths: Dict[str, Any] = {"type": spec.get("type", "bedmethyl")}
        try:
            for label in ("bedmethyl", "regions"):
                url = spec[label]
                destination = dataset_dir / Path(url).name
                dataset_paths[label] = _download_file(url, destination)
        except Exception as exc:  # pragma: no cover - network dependent
            failures[dataset_key] = exc
            continue

        datasets[dataset_key] = dataset_paths

    if not datasets:
        reason = "; ".join(f"{key}: {exc}" for key, exc in failures.items()) or "unknown error"
        pytest.skip(f"Unable to download remote datasets: {reason}")

    return datasets


@pytest.fixture(scope="session")
def remote_bedtables(remote_dataset_paths: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, BedTable]]:
    """
    Loads remote datasets into BedTable objects once.
    """
    out: Dict[str, Dict[str, BedTable]] = {}
    for key, ds in remote_dataset_paths.items():
        out[key] = {
            "bedmethyl": BedTable.from_path(ds["bedmethyl"]),
            "regions": BedTable.from_path(ds["regions"]),
        }
    return out


@pytest.fixture()
def mpl_agg(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Force matplotlib headless backend for plot tests.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("DISPLAY", "")


@pytest.fixture()
def run_main_no_parallel(monkeypatch: pytest.MonkeyPatch):
    """
    Run centrodip.main.main() inline, forcing no multiprocessing.
    Also patches ProcessPoolExecutor so code paths using it still work.
    """
    import concurrent.futures

    class _InlineExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            f = concurrent.futures.Future()
            try:
                f.set_result(fn(*args, **kwargs))
            except BaseException as e:
                f.set_exception(e)
            return f

    def _run(argv):
        import centrodip.main as cmain

        # patch the executor used by centrodip.main
        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)

        # run with patched argv
        monkeypatch.setattr(sys, "argv", ["centrodip"] + [str(a) for a in argv])
        cmain.main()

    return _run