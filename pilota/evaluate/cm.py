#!/usr/bin/env python3
import argparse
import difflib
from pathlib import Path

from pilota.evaluate.schema import Result


def operation(path_in: Path, path_out: Path) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            result = Result.parse_raw(line)

            a = "__@@__".join(["__".join(v) for v in result.target_predicted[-1].scuds_tokenized])
            b = "__@@__".join(["__".join(v) for v in result.target_golds_tokenized])

            s = difflib.SequenceMatcher(
                a=a,
                b=b,
            )

            for op, a_0, a_1, b_0, b_1 in s.get_opcodes():
                #                 if op == "replace":
                #                     continue
                #                 elif op == "insert":
                #                     continue
                #                 elif op == "delete":
                #                     continue
                if op == "equal":
                    continue

                _gold: str = a[a_0:a_1].replace("__", "")
                _predicted: str = b[b_0:b_1].replace("__", "")

                outf.write(f"{op}\t{_predicted}\t{_gold}\n")


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
