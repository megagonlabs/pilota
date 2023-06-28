#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Union

import numpy as np

from pilota.mull.mull import Mull
from pilota.mull.train import DEFAULT_SEP


def predict(
    *,
    path_model: Path,
    path_input: Path,
    path_output: Path,
    show_json: bool,
    batch_size: int,
    separator: str,
    pair: bool,
):
    def get_input(inf, batch_size: int) -> Iterator[Union[list[str], list[list[str]]]]:
        found_sep: bool = False
        rets: Union[list[str], list[list[str]]] = []
        while True:
            line: str = inf.readline()
            if len(line) == 0:
                break
            text: str = line.strip()

            if not pair:
                rets.append(text)
                if not found_sep and separator in text:
                    sys.stderr.write(f"[Warning] Found separator '{separator}' but --pair option is not used.\n")
                    found_sep = True
            else:
                vs: list[str] = text.split(separator, maxsplit=1)
                assert len(vs) == 2
                rets.append(vs)  # type: ignore

            if len(rets) == batch_size:
                yield rets
                rets = []
        if len(rets) != 0:
            yield rets

    mull = Mull.from_pretrained(path_model)
    with path_output.open("w") as outf, path_input.open() as inf:
        for _in in get_input(inf, batch_size):
            predictions = mull.predict(inputs=_in)

            for text, prediction_ in zip(_in, predictions):
                if show_json:
                    kvs = {}
                    for lab, val in zip(mull.labels, prediction_):
                        kvs[lab] = float(val)
                    outf.write(f"{json.dumps(kvs, sort_keys=True, ensure_ascii=False)}\n")
                else:
                    outs = []
                    for idx in np.argsort(-prediction_)[:5]:
                        outs.append(f"{mull.labels[idx]} ({prediction_[idx]*100:0.3f})")
                    out_text: str
                    if isinstance(text, list):
                        out_text = json.dumps(text, ensure_ascii=False)
                    else:
                        out_text = text
                    outf.write(f'{out_text}\n\t{", ".join(outs)}\n\n')


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--model", "-m", type=Path, required=True)
    oparser.add_argument("--json", action="store_true")
    oparser.add_argument("--bs", "--batch_size", type=int, default=1)
    oparser.add_argument("--separator", default=DEFAULT_SEP)
    oparser.add_argument("--pair", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()

    predict(
        path_model=opts.model,
        path_input=opts.input,
        path_output=opts.output,
        show_json=opts.json,
        batch_size=opts.bs,
        separator=opts.separator,
        pair=opts.pair,
    )


if __name__ == "__main__":
    main()
