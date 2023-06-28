#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def operation(path_in: Path, path_out: Path, switch: bool) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            items1 = line[:-1].split("\t")
            #             result1 = items1[2]
            score1 = json.loads(items1[-1])[1]

            items2 = inf.readline().split("\t")
            #             result2 = items2[2]
            score2 = json.loads(items2[-1])[1]

            #             if result1 == result2 == 'E':
            if score1 >= 0.8 and score2 >= 0.8:
                if not switch:
                    outf.write(f"{items1[0]}\t{items1[1]}\t{score1}\t{score2}\n")
            else:
                if switch:
                    outf.write(f"{items1[0]}\t{items1[1]}\t{score1}\t{score2}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--switch", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(opts.input, opts.output, opts.switch)


if __name__ == "__main__":
    main()
