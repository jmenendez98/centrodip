from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict
from urllib.error import ContentTooShortError, HTTPError, URLError
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centrodip.parser import Parser

REMOTE_DATASETS: Dict[str, Dict[str, str]] = {
    "chm13_chr1": {
        "bedmethyl": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/CHM13_ONT_dorado.5mC_pileup.bed",
        "regions": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/CHM13v2.0_ONT_5mC.chr1.bed",
    },
    "hg002_chrX": {
        "bedmethyl": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/HG002v1.1_Q100_ONT_5mC.H1L.chrX.bed",
        "regions": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/hg002v1.1.cenSatv2.0.chrX.H1L.bed",
    },
    "hg002_chrY": {
        "bedmethyl": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/HG002v1.1_Q100_ONT_5mC.chrY.bed",
        "regions": "https://public.gi.ucsc.edu/~jmmenend/.centrodip_test_data/hg002v1.1.cenSatv2.0.chrY.H1L.bed",
    },
}

EXPECTED_CHROMS: Dict[str, str] = {
    "chm13_chr1": "chr1",
    "hg002_chrX": "chrX_MATERNAL",
    "hg002_chrY": "chrY_PATERNAL",
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
def bed_parser() -> Parser:
    return Parser(
        mod_code="m",
        bedgraph=False,
    )


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def remote_dataset_paths(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Dict[str, Path]]:
    base_dir = tmp_path_factory.mktemp("centrodip_remote_data")
    datasets: Dict[str, Dict[str, Path]] = {}
    failures: Dict[str, Exception] = {}

    for dataset_key, urls in REMOTE_DATASETS.items():
        dataset_dir = base_dir / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_paths: Dict[str, Path] = {}
        try:
            for label, url in urls.items():
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