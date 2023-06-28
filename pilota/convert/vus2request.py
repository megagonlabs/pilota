#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Literal

from pilota.schema import Utterance


def rename_username(name: str) -> Literal["user", "agent"]:
    if name == "user":
        return "user"
    elif name == "agent":
        return "agent"
    raise KeyError(name)


def operation(path_in: Path, path_out: Path, is_sentence: bool) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            vus = json.loads(line)
            uttrs = []
            for uttr in vus["utterances"]:
                uttr_obj = Utterance(
                    name=rename_username(uttr["name"]),
                    text=uttr["text"],
                )
                uttrs.append(uttr_obj)

            if is_sentence:
                req = {
                    "id": f'{vus["docid"]["id"]}.0',
                    "context": None,
                    "utterance": None,
                    "sentences": [uttr.text for uttr in uttrs],
                }
                outf.write(json.dumps(req, ensure_ascii=False))
                outf.write("\n")
                continue

            for idx, uttr in enumerate(uttrs):
                if uttr.name == "agent":
                    continue
                ctx = []
                if idx >= 1:
                    for _u in uttrs[:idx]:
                        ctx.append(_u.dict())
                req = {
                    "id": f'{vus["docid"]["id"]}.{idx}',
                    "context": ctx,
                    "utterance": uttr.text,
                    "sentences": None,
                }
                outf.write(json.dumps(req, ensure_ascii=False))
                outf.write("\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--sentence", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(opts.input, opts.output, opts.sentence)


if __name__ == "__main__":
    main()
