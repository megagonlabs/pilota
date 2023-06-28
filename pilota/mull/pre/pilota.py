#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from pilota.mull.train import DEFAULT_SEP


def get_purpose(idx, divide: int = 10):
    purpose: str = "train"
    if idx % divide == 0:
        purpose = "dev"
    elif idx % divide == 1:
        purpose = "test"
    return purpose


def operation(path_in: Path, path_out: Path) -> None:
    idx: int = 1
    with path_in.open() as inf, path_out.open("w") as outf:
        r = csv.reader(inf)
        _ = next(r)
        for cnt, items in enumerate(r):
            text_in = items[1]
            gold = items[2]
            ngs = [it for it in items[3:6]]
            purpose: str = get_purpose(cnt)

            # 未アノテーションが1つでもあれば
            if len(list(filter(lambda x: len(x) == 0, ngs))) > 0:
                continue

            outf.write(f"ex{idx}\t{text_in}{DEFAULT_SEP}{gold}\tOK\t{purpose}\n")
            idx += 1
            for ng in ngs:
                outf.write(f"ex{idx}\t{text_in}{DEFAULT_SEP}{ng}\tN/A\t{purpose}\n")
                idx += 1


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    assert opts.input.is_dir()
    operation(opts.input.joinpath("neg.csv"), opts.output)


if __name__ == "__main__":
    main()
