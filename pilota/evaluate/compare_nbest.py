#!/usr/bin/env python3

import argparse
from pathlib import Path

from pilota.evaluate.schema import Result, ScudsResult


def operation(path_in: Path, path_out: Path) -> None:
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            r: Result = Result.parse_raw(line)
            reranked: ScudsResult = r.target_predicted[-1]
            orig: ScudsResult = r.target_predicted[0]

            delta_rouge_l: float = reranked.rouge_score.rouge_l - orig.rouge_score.rouge_l

            assert reranked.score is not None
            assert orig.score is not None

            outf.write(
                f"{r.sid}\t{r.source_text}\t{r.target_gold}\t"
                f"{reranked.text}\t{orig.text}"
                f"\t{delta_rouge_l:.4}"
                f"\t{reranked.rouge_score.rouge_l:.4}\t{orig.rouge_score.rouge_l:.4}"
                f"\t{reranked.score:.4}\t{orig.score:.4}\n"
            )


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
