#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Final, Optional

from pilota.const.metachar import MARK_CONTEXT, NONE, PHRASE_SEP, SEP
from pilota.mull.mull import Mull
from pilota.schema import ScorerConfig

LABEL_OK: str = "OK"
CONFIG_FILE_NAME: Final[str] = "scorer.config.json"


def get_scorer_input(*, src: str, target: str, context_separator: str) -> str:
    return target.replace(PHRASE_SEP, "") + context_separator + src.replace(PHRASE_SEP, "")


def score_details2score(
    *,
    scores_detail: dict[str, float],
    original_rank: int,
) -> float:
    final_score: float = 0.0
    score_ok: float = scores_detail.get(LABEL_OK, 0.0)
    if score_ok > 0.7:
        final_score += 0.4

    original_rank = max(original_rank, 0)
    final_score += 0.3 / (original_rank + 1.0)

    final_score += 0.3 * score_ok

    return final_score


class Scorer:
    _mdl: Mull
    config: ScorerConfig

    def __init__(self, path_model: Path):
        self._mdl = Mull.from_pretrained(path_model)
        self.config = ScorerConfig.parse_file(path_model.joinpath(CONFIG_FILE_NAME))

    @property
    def labels(self):
        return self._mdl.labels

    def calc(
        self,
        sources: list[str],
        scuds_list: list[list[str]],
        in_max_length: Optional[int],
    ) -> tuple[list[float], list[dict[str, float]]]:
        if isinstance(in_max_length, int):
            if in_max_length <= 0:
                in_max_length = None

        inputs = []
        nbest: int = int(len(scuds_list) / len(sources))
        for k, src in enumerate(sources):
            for j in range(nbest):
                scuds: list[str] = scuds_list[k * nbest + j]
                if len(scuds) == 0:
                    inputs.append(
                        get_scorer_input(
                            src=src,
                            target=NONE,
                            context_separator=self.config.context_separator,
                        )
                    )
                else:
                    inputs.append(
                        get_scorer_input(
                            src=src,
                            target=SEP.join(scuds),
                            context_separator=self.config.context_separator,
                        )
                    )
        ret0 = [
            v
            for v in self._mdl.predict(
                inputs=inputs,
                in_max_length=in_max_length,
            )
        ]

        scores_detail = [{k: float(v) for k, v in zip(self._mdl.labels, item)} for item in ret0]
        scores = [
            score_details2score(
                scores_detail=sd,
                original_rank=idx,
            )
            for idx, sd in enumerate(scores_detail)
        ]
        return scores, scores_detail

    def rerank(
        self,
        *,
        scuds_list: list[list[str]],
        original_ranks: list[int],
        scores: list[float],
        scores_detail: list[dict[str, float]],
        num_sentences: int,
        force_none_lowrank: bool,
    ):
        assert len(scuds_list) == len(scores_detail) == len(original_ranks)
        nbest: int = int(len(scuds_list) / num_sentences)

        def my_new_score(scuds, score: float):
            if force_none_lowrank and len(scuds) == 0:
                return -999
            return score

        for src_idx in range(num_sentences):
            _from: int = src_idx * nbest
            _to: int = (src_idx + 1) * nbest
            new_sl = []
            new_scores = []
            new_scores_detail = []
            new_original_ranks = []
            part_sl = scuds_list[_from:_to]
            part_scores = scores[_from:_to]
            part_scores_detail = scores_detail[_from:_to]
            for scuds, score, part_score_detail, original_rank in sorted(
                zip(
                    part_sl,
                    part_scores,
                    part_scores_detail,
                    original_ranks,
                ),
                key=lambda x: my_new_score(x[0], x[1]),
                reverse=True,
            ):
                new_sl.append(scuds)
                new_scores.append(score)
                new_scores_detail.append(part_score_detail)
                new_original_ranks.append(original_rank)
            scuds_list[_from:_to] = new_sl
            scores[_from:_to] = new_scores
            scores_detail[_from:_to] = new_scores_detail
            original_ranks[_from:_to] = new_original_ranks


def get_opts() -> argparse.ArgumentParser:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--model", "-m", type=Path, required=True)
    return oparser


def main() -> None:
    opts = get_opts().parse_args()
    scorer = Scorer(path_model=opts.model)

    path_in: Path = opts.input
    path_out: Path = opts.output
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            items: list[str] = line[:-1].split(MARK_CONTEXT, maxsplit=1)
            assert len(items) == 2

            _, score_details_list = scorer.calc(
                sources=[items[1]],
                scuds_list=[items[0].split(SEP)],
                in_max_length=None,
            )
            outf.write(json.dumps(score_details_list[0]))
            outf.write("\n")


if __name__ == "__main__":
    main()
