#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

from pilota.mull.train import DEFAULT_SEP


def rhr_pn(path_in: Path, path_out: Path) -> None:
    sent2vals: dict[str, dict[str, str]] = {}
    for key, fname in {"sentiment": "pn", "rhr": "rhr"}.items():
        with path_in.joinpath(f"{fname}.tsv").open() as inf:
            for line in inf:
                items = line.split("\t")
                sentence: str = items[2]
                label: str = items[1]

                tmp = sent2vals.get(sentence, {})
                tmp[key] = label
                sent2vals[sentence] = tmp

    with path_out.open("w") as outf:
        keys = list(sent2vals.keys())
        random.shuffle(keys)
        for idx, sent in enumerate(keys):
            vals = sent2vals[sent]
            mylabels: list[str] = []
            if vals["sentiment"] == "1":
                mylabels.append("sentiment")
            elif vals["sentiment"] == "0":
                mylabels.append("sentiment@@0.5")

            if vals["rhr"] == "1":
                mylabels.append("rhr")

            if len(mylabels) == 0:
                mylabels = ["N/A"]

            purpose: str = "train"
            if idx % 10 == 0:
                purpose = "dev"
            elif idx % 10 == 1:
                purpose = "test"

            outf.write(f'example_{idx:05}\t{sent}\t{",".join(mylabels)}\t{purpose}\n')


def rte(path_in: Path, path_out: Path) -> None:
    with path_out.open("w") as outf:
        for inpath in path_in.iterdir():
            if not inpath.name.startswith("rte."):
                continue
            with inpath.open() as inf:
                for line in inf:
                    items = line[:-1].split("\t")
                    if items[1] == "1":
                        label = "E"
                    else:
                        label = "N/A"
                    outf.write(f"{items[0]}\t{items[2]}{DEFAULT_SEP}{items[3]}\t{label}\t{items[-1]}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, required=True)
    oparser.add_argument("--rte", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()

    assert opts.input.is_dir()
    opts.output.parent.mkdir(exist_ok=True, parents=True)
    random.seed(42)
    if opts.rte:
        rte(opts.input, opts.output)
    else:
        rhr_pn(opts.input, opts.output)


if __name__ == "__main__":
    main()
