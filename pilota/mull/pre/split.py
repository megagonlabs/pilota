#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PURPOSES: list[str] = ["test", "train", "dev"]


def operation(
    path_in_list: list[Path],
    path_out: Path,
    idx_text: int,
    idx_label: int,
    idx_purpose: int,
    purpose: str,
    path_labels: Optional[Path],
) -> None:
    path_out.parent.mkdir(parents=True, exist_ok=True)
    labels = set()

    input_data = []
    for path_in in path_in_list:
        with path_in.open() as inf:
            input_data += inf.readlines()

    found: bool = False
    with path_out.open("w") as outf:
        for line in input_data:
            items = line[:-1].split("\t")
            mylabels = []
            for t in items[idx_label].split(","):
                if t == "N/A":
                    continue
                mylabels.append(t)

            assert items[idx_purpose] in PURPOSES, f"Invalid purpose: {items[idx_purpose]} in {items}"
            if items[idx_purpose] == purpose:
                for t in mylabels:
                    labels.add(t.split("@@")[0])
                outf.write(
                    f"{items[0]}\t{items[idx_text]}" f"\t{json.dumps(mylabels, ensure_ascii=False, sort_keys=True)}\n"
                )
                found = True
    if not found:
        sys.stderr.write(f"Warning: No {purpose} found.\n")

    if path_labels:
        with path_labels.open("w") as outf:
            for t in sorted(list(labels)):
                outf.write(f"{t}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True, action="append")
    oparser.add_argument("--output", "-o", type=Path, required=True)
    oparser.add_argument("--index-text", type=int, default=1)
    oparser.add_argument("--index-label", type=int, default=2)
    oparser.add_argument("--index-purpose", type=int, default=3)
    oparser.add_argument("--purpose", choices=PURPOSES, required=True)
    oparser.add_argument("--labels", type=Path)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(
        opts.input,
        opts.output,
        idx_text=opts.index_text,
        idx_purpose=opts.index_purpose,
        idx_label=opts.index_label,
        purpose=opts.purpose,
        path_labels=opts.labels,
    )


if __name__ == "__main__":
    main()
