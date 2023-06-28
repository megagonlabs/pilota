#!/usr/bin/env python3

import argparse
from pathlib import Path

from asdc.schema.example import Example


def joint(path_in_list: list[Path], path_out: Path) -> None:
    ids = set()
    with path_out.open("w") as outf:
        for path_in in path_in_list:
            with path_in.open() as inf:
                for line in inf:
                    myid = line.strip().split("\t")[0]
                    if myid in ids:
                        raise KeyError
                    ids.add(myid)
                    outf.write(line)


def split(path_in: Path, path_out: Path, path_src: Path, path_ref: Path) -> None:
    ids: set[str] = set()
    with path_src.open() as sf:
        for line in sf:
            ex: Example = Example.parse_raw(line)
            ids.add(ex.sid.id)

    out_one: bool = False
    with path_in.open() as inf, path_ref.open() as rf, path_out.open("w") as outf:
        for rline, line in zip(rf, inf):
            myid = rline.strip().split("\t")[0]
            if myid not in ids:
                continue
            outf.write(line)
            out_one = True
    if not out_one:
        raise KeyError("Nothing was output")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, action="append")
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--src", "-s", type=Path)
    oparser.add_argument("--ref", "-r", type=Path)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    if opts.src:
        assert len(opts.input) == 1
        assert opts.ref is not None
        split(opts.input[0], opts.output, opts.src, opts.ref)
    else:
        joint(opts.input, opts.output)


if __name__ == "__main__":
    main()
