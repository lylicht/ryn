"""Keep Hugging Face dataset caches local and avoid Windows URL/drive confusion."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def is_remote_uri(path: str | Path) -> bool:
    scheme = urlparse(str(path)).scheme
    return bool(scheme) and scheme.lower() != "file" and not (len(scheme) == 1 and scheme.isalpha())


def configure_hf_dataset_runtime(cache_root: str | Path) -> None:
    root = Path(cache_root).expanduser().resolve()
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")

    from datasets import data_files
    from datasets.utils import file_utils

    # datasets 4.x tests ``hf:/`` with Windows' mount check, which can stall on mapped drives.
    file_utils.is_remote_url = is_remote_uri
    file_utils.is_local_path = lambda path: not is_remote_uri(path)
    data_files.is_local_path = lambda path: not is_remote_uri(path)
