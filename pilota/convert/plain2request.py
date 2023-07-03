#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from pilota.schema import Request, Utterance


def get_context(items) -> list[Utterance]:
    ret: list[Utterance] = []
    for idx in range(-len(items), 0):
        name: str = "agent"
        if idx % 2 == 0:
            name = "user"
        ret.append(
            Utterance(
                name=name,
                text=items[idx],
            )
        )
    return ret


def operation(path_in: Path, path_out: Path) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for items in csv.reader(inf, delimiter="\t"):
            uttr: str = ""
            if len(items) == 0:
                ctx = None
            elif len(items) == 1:
                ctx = None
                uttr = items[0]
            else:
                ctx = get_context(items[:-1])
                uttr = items[-1]

            req = Request(
                context=ctx,
                utterance=uttr,
                sentences=None,
            )
            outf.write(req.json(ensure_ascii=False))
            outf.write("\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(opts.input, opts.output)


if __name__ == "__main__":
    main()
