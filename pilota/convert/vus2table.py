#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

from pilota.predict import ResultForSentence


def operation(path_in: Path, path_result: Path, path_out: Path) -> None:
    with path_result.open() as in_result, path_out.open("w") as outf, path_in.open() as inf:
        w = csv.writer(outf)

        for line in inf:
            vus = json.loads(line)
            w.writerow([vus["meta"]["id"]["id"]])

            for uttr in vus["utterances"]:
                if uttr["name"] == "agent":
                    w.writerow(["", uttr["text"]])
                else:
                    ds = json.loads(next(in_result))
                    for d in ds:
                        rs = ResultForSentence.parse_obj(d)
                        w.writerow(["(user)", rs.sentence, "\n".join(rs.scuds_nbest[0])])
            w.writerow([])


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--result", "-r", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(opts.input, opts.result, opts.output)


if __name__ == "__main__":
    main()
