#!/usr/bin/env python3

from pathlib import Path

from huggingface_hub import snapshot_download


def get_real_model_path(path_model_str: str) -> Path:
    if path_model_str.startswith("megagonlabs/"):
        return Path(
            snapshot_download(
                repo_id=path_model_str,
            )
        )
    return Path(path_model_str)
