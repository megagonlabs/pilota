#!/usr/bin/env python3

import argparse
from pathlib import Path

from pilota.evaluate.schema import Result


def operation(*, path_old: Path, path_new: Path, path_out: Path) -> None:
    examples0: dict[str, Result] = {}
    examples1: dict[str, Result] = {}

    lid: int = 0
    with path_old.open() as inf:
        for line in inf:
            lid += 1
            r = Result.parse_raw(line)
            examples0[r.sid] = r
    with path_new.open() as inf:
        for line in inf:
            r = Result.parse_raw(line)
            examples1[r.sid] = r

    assert len(examples0) == len(examples1) == lid

    with path_out.open("w") as outf:
        for sid, e0 in examples0.items():
            e1 = examples1[sid]
            diff = e1.target_predicted[-1].rouge_score.rouge_l - e0.target_predicted[-1].rouge_score.rouge_l
            outf.write(
                f"{sid}\t{diff}\t{e0.target_predicted[-1].rouge_score.rouge_l}"
                f"\t{e1.target_predicted[-1].rouge_score.rouge_l}"
            )
            outf.write(f"\t{e0.input}\t{e0.target_gold}\t{e0.target_predicted}\t{e1.target_predicted}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--new", type=Path, required=True)
    oparser.add_argument("--old", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(path_old=opts.old, path_new=opts.new, path_out=opts.output)


if __name__ == "__main__":
    main()
