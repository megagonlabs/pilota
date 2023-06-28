#!/usr/bin/env python3

import argparse
from pathlib import Path


def operation(path_in: Path, path_out: Path) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            outf.write(line)


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
