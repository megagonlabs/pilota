#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

from pilota.mull.train import DEFAULT_SEP


def get_purpose(idx, divide: int = 10):
    purpose: str = "train"
    if idx % divide == 0:
        purpose = "dev"
    elif idx % divide == 1:
        purpose = "test"
    return purpose


def operation_rte(path_in: Path, path_out: Path) -> None:
    label2labels: dict[str, list[str]] = {
        "=": ["="],
        "AB": [],
        "BA": [],
        "N": ["N/A"],
    }

    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            items = line[:-1].split("\t")
            label: str = items[3]
            if label == "xx":
                continue

            mylabels: list[str] = label2labels[label]
            if len(mylabels) == 0:
                continue

            myid: str = items[0]
            text_a = items[1]
            text_b = items[2]
            purpose: str = items[4]

            outf.write(f'{myid}\t{text_a}{DEFAULT_SEP}{text_b}\t{",".join(mylabels)}\t{purpose}\n')


def operation_eq(path_in: Path, path_out: Path) -> None:
    label2labels: dict[str, list[str]] = {
        "=": ["="],
        "N": ["N/A"],
    }

    idx: int = 0
    path_out.parent.mkdir(exist_ok=True, parents=True)
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            items = line[:-1].split("\t")
            label: str = items[3]
            if label == "xx":
                continue

            dist = json.loads(items[-1])
            assert sum(v for v in dist.values()) == 5
            if dist.get("N", 0) == 3 or dist.get("=", 0) == 3:
                continue

            mylabels: list[str] = label2labels[label]
            if len(mylabels) == 0:
                continue

            myid = items[0]
            text_a = items[1]
            text_b = items[2]
            purpose: str = items[4]

            outf.write(f'{myid}\t{text_a}{DEFAULT_SEP}{text_b}\t{",".join(mylabels)}\t{purpose}\n')
            idx += 1


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--eq", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    random.seed(42)

    assert opts.input.is_dir()

    opts.output.parent.mkdir(exist_ok=True, parents=True)
    if opts.eq:
        operation_eq(opts.input.joinpath("eq.tsv"), opts.output)
    else:
        operation_rte(opts.input.joinpath("rte.tsv"), opts.output)


if __name__ == "__main__":
    main()
