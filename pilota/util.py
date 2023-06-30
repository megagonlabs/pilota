#!/usr/bin/env python3

from pathlib import Path

from huggingface_hub import snapshot_download


def get_real_model_path(
    path_model_str: str,
    *,
    check_model_update: bool = False,
) -> Path:
    if path_model_str.startswith("megagonlabs/"):
        if not check_model_update:
            try:
                return Path(
                    snapshot_download(
                        repo_id=path_model_str,
                        local_files_only=True,
                    )
                )
            except FileNotFoundError:
                pass
        return Path(
            snapshot_download(
                repo_id=path_model_str,
            )
        )
    return Path(path_model_str)
