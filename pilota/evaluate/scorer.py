#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from pilota.mull.pre.evaluate import operation as gen_txt
from pilota.mull.pre.evaluate import score


def get_group_name(name: str) -> str:
    pre0 = "correctness_labeled."
    if name.startswith(pre0):
        name = name[len(pre0) :]

    if name.startswith("candidate"):
        name = ".".join(name.split(".")[2:])

    return ".".join(name.split(".")[:2])


def operation(
    *,
    path_in: Path,
    path_gold: Path,
    path_label: Path,
    path_out: Path,
) -> None:
    gname2og = {}

    fname_test: str = "test.tsv"
    fname_prediction: str = "prediction.jsonl"
    fname_prediction_txt: str = "prediction.txt"
    with path_in.open() as inf, path_gold.open() as gf:
        gf2 = csv.reader(gf, delimiter="\t")
        for gold_items, predicted_line in zip(gf2, inf):
            gname: str
            for gname in [get_group_name(gold_items[0]), "_all"]:
                _tmp = gname2og.get(gname)
                if _tmp is None:
                    odir = path_out.joinpath(gname)
                    odir.mkdir(exist_ok=True, parents=True)
                    of1 = odir.joinpath(fname_test).open("w")
                    of2 = odir.joinpath(fname_prediction).open("w")
                    gname2og[gname] = (of1, of2)
                else:
                    of1, of2 = _tmp

                of1.write("\t".join(gold_items))
                of1.write("\n")
                of2.write(predicted_line)

        for gname, _tmp in gname2og.items():
            _tmp[0].close()
            _tmp[1].close()

            ptxt = path_out.joinpath(gname, fname_prediction_txt)
            gen_txt(
                path_in=path_out.joinpath(gname, fname_prediction),
                path_out=ptxt,
                th=0.5,
            )
            score(
                path_in=ptxt,
                path_gold=path_out.joinpath(gname, fname_test),
                path_label=path_label,
                path_out=path_out.joinpath(gname, "stat.tsv"),
            )


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--gold", "-g", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, required=True)
    oparser.add_argument("--label", "-l", type=Path, required=True)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(
        path_in=opts.input,
        path_gold=opts.gold,
        path_out=opts.output,
        path_label=opts.label,
    )


if __name__ == "__main__":
    main()
