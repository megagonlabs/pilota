#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from pydantic import BaseModel


def operation(path_in: Path, path_out: Path, th: float) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            out = []
            for k, v in json.loads(line).items():
                if v > th:
                    out.append(k)
            outf.write(json.dumps(sorted(out), ensure_ascii=False))
            outf.write("\n")


class Scorer(BaseModel):
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self):
        denominator: int = self.tp + self.fp
        if denominator == 0:
            return float("nan")
        return self.tp / float(denominator)

    @property
    def recall(self):
        denominator: int = self.tp + self.fn
        if denominator == 0:
            return float("nan")
        return self.tp / float(denominator)

    @property
    def f1(self):
        denominator: float = self.precision + self.recall
        if denominator == 0.0:
            return float("nan")
        return 2 * (self.precision * self.recall) / denominator


def score(path_in: Path, path_gold: Path, path_label: Path, path_out: Path) -> None:
    name2scorer = {}

    with path_label.open() as lf:
        for line in lf:
            name2scorer[line.strip()] = Scorer()

    with path_in.open() as inf, path_gold.open() as goldf, path_out.open("w") as outf:
        for line, goldline in zip(inf, goldf):
            sp = json.loads(line)
            g = json.loads(goldline.split("\t")[-1])
            for lb in name2scorer.keys():
                if lb in sp:
                    if lb in g:
                        name2scorer[lb].tp += 1
                    else:
                        name2scorer[lb].fp += 1
                else:
                    if lb in g:
                        name2scorer[lb].fn += 1
                    else:
                        name2scorer[lb].tn += 1

        for name, scorer in name2scorer.items():
            outf.write(f"{name}\t{scorer.json()}\t")
            outf.write(f"{scorer.precision:0.4f}\t{scorer.recall:0.4f}\t{scorer.f1:0.4f}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout")
    oparser.add_argument("--th", type=float, default=0.5)
    oparser.add_argument("--score", action="store_true")
    oparser.add_argument("--gold", "-g", type=Path, required=False)
    oparser.add_argument("--label", "-l", type=Path, required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    if opts.score:
        score(opts.input, opts.gold, opts.label, opts.output)
    else:
        operation(opts.input, opts.output, opts.th)


if __name__ == "__main__":
    main()
