#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def operation(path_in: Path, path_out: Path, topic: str, th: float) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            text, json_t = line[:-1].split("\t")
            data = json.loads(json_t)
            if data[topic] > th:
                outf.write(f"{text}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--topic", required=True)
    oparser.add_argument("--th", type=float, required=True)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(opts.input, opts.output, opts.topic, opts.th)


if __name__ == "__main__":
    main()
