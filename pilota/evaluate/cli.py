#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional

from pilota.const.metachar import NONE, SEP
from pilota.predict import ResultForSentence

try:
    from sumeval.metrics.rouge import RougeCalculator
except ImportError:
    raise Exception("You need to install the extra 'train'")

from pydantic import BaseModel

from pilota.const.metachar import TARGET_END, TARGET_START
from pilota.evaluate.calc import get_rouge_score, get_tokenized
from pilota.evaluate.schema import Result, ScudsResult
from pilota.predict import Predictor


class Predicted(BaseModel):
    text: str
    score: Optional[float]
    score_detail: Optional[dict[str, float]]


def evaluate_one(
    *,
    sid: str,
    src: str,
    target_gold: str,
    n2target_predicted_text: dict[int, str],
    n2target_predicted_score_detail: dict[int, Optional[dict[str, float]]],
    rouge: RougeCalculator,
) -> Iterator[Result]:
    items_gold: list[str] = Predictor.decode_to_scuds(target_gold)
    items_gold_tokenized: list[list[str]] = get_tokenized(items_gold, rouge)

    source_text: str = src.split(TARGET_START, maxsplit=2)[-1].split(TARGET_END, maxsplit=2)[0]
    source_text_tokenized: list[str] = rouge._lang.tokenize(source_text)

    target_predicted_result: dict[int, ScudsResult] = {}

    for n, target_predicted_text in n2target_predicted_text.items():
        items_system: list[str] = Predictor.decode_to_scuds(target_predicted_text)
        items_system_tokenized: list[list[str]] = get_tokenized(items_system, rouge)
        score: Optional[dict[str, float]] = n2target_predicted_score_detail[n]

        rouge_score = get_rouge_score(
            items_gold_tokenized,
            items_system_tokenized,
            rouge,
        )

        target_predicted_result[n] = ScudsResult(
            text=target_predicted_text,
            score=score,
            scuds_tokenized=items_system_tokenized,
            rouge_score=rouge_score,
        )

    yield Result(
        sid=sid,
        input=src,
        source_text=source_text,
        source_text_tokenized=source_text_tokenized,
        target_gold=target_gold,
        target_golds_tokenized=items_gold_tokenized,
        target_predicted=target_predicted_result,
    )


def evaluate_from_file(
    *,
    path_gold: Path,
    path_in: Path,
    path_out: Path,
    path_txt_dir: Path,
    nbest_idx_list: list[int],
) -> None:
    sids: list[str] = []
    sources: list[str] = []
    target_golds: list[str] = []

    with path_gold.open() as gf:
        for line in gf:
            items = line[:-1].split("\t")
            sids.append(items[0])
            sources.append(items[1])
            target_golds.append(items[2])

    n_2_target_predicteds: dict[int, list[Predicted]] = {}

    path_txt_dir.mkdir(exist_ok=True, parents=True)
    for nbest_idx in nbest_idx_list:
        n_2_target_predicteds[nbest_idx] = []  # initialize
        path_txt_file = path_txt_dir.joinpath(f"{nbest_idx}.txt")
        with path_in.open() as inf, path_txt_file.open("w") as out_txtf:
            for line in inf:
                ds = json.loads(line)
                assert len(ds) == 1
                rfs = ResultForSentence.parse_obj(ds[0])
                idx: int = 0
                if nbest_idx >= 0:
                    idx = rfs.original_ranks.index(nbest_idx)
                scuds = rfs.scuds_nbest[idx]
                scuds_line: str = SEP.join(scuds)
                if len(scuds_line) == 0:
                    scuds_line = NONE
                n_2_target_predicteds[nbest_idx].append(
                    Predicted(
                        text=scuds_line,
                        score=None if rfs.scores is None else rfs.scores[idx],
                        score_detail=None if rfs.scores_detail is None else rfs.scores_detail[idx],
                    )
                )
                out_txtf.write(f"{scuds_line}\n")

    with path_out.open("w") as outf:
        for result in evaluate(sids, sources, target_golds, n_2_target_predicteds):
            outf.write(result.model_dump_json())
            outf.write("\n")


def evaluate(
    sids: list[str],
    sources: list[str],
    target_golds: list[str],
    n_2_target_predicteds: dict[int, list[Predicted]],
) -> Iterator[Result]:
    rouge = RougeCalculator(lang="ja", stopwords=False, stemming=False)

    for target_predicteds in n_2_target_predicteds.values():
        assert len(sids) == len(sources) == len(target_golds) == len(target_predicteds)

    for idx, (sid, src, target_gold) in enumerate(zip(sids, sources, target_golds)):
        n2target_predicted_text: dict[int, str] = {}
        n2target_predicted_score_detail: dict[int, Optional[dict[str, float]]] = {}
        for n, target_predicteds in n_2_target_predicteds.items():
            n2target_predicted_text[n] = target_predicteds[idx].text
            n2target_predicted_score_detail[n] = target_predicteds[idx].score_detail

        for result in evaluate_one(
            sid=sid,
            src=src,
            target_gold=target_gold,
            n2target_predicted_text=n2target_predicted_text,
            n2target_predicted_score_detail=n2target_predicted_score_detail,
            rouge=rouge,
        ):
            yield result


def stat(path_in_list: list[Path], path_out: Path) -> None:
    def sid2groupname(sid: str) -> str:
        if sid.startswith("asdc.v4"):
            return "main"
        items = sid.split(".")
        if len(items) >= 3:
            return items[2]
        return "unknown"

    vals: dict[str, dict[str, list[float]]] = {}
    for path_in in path_in_list:
        with path_in.open() as inf:
            for line in inf:
                result = Result.parse_raw(line)
                myname = sid2groupname(result.sid)
                for name in [myname, "ALL"]:
                    if name not in vals:
                        vals[name] = {}

                    for n, target_predicted in result.target_predicted.items():
                        for sc in [target_predicted.rouge_score]:
                            for score_name in sc.__fields__.keys():
                                my_score_name: str = f"{n}__{score_name}"
                                if my_score_name not in vals[name]:
                                    vals[name][my_score_name] = []
                                vals[name][my_score_name].append(sc.__dict__[score_name])

    with path_out.open("w") as outf:
        for name, kvs in sorted(vals.items()):
            for k, vs in sorted(kvs.items()):
                final: float = sum(vs) / len(vs)
                outf.write(f"{name}\t{k}\t{final:.3}\t{len(vs)}\n")


def csv_output(path_in: Path, path_out: Path) -> None:
    import csv

    with path_in.open() as inf, path_out.open("w") as outf:
        cof = csv.writer(outf)

        cof.writerow(["SID", "Input", "GOLD", "system", "rouge_l"])

        for line in inf:
            result = Result.parse_raw(line)
            items = [
                result.sid,
                result.input,
                result.target_gold,
                result.target_predicted[-1].text,
                result.target_predicted[-1].rouge_score.rouge_l,
            ]
            cof.writerow(items)


#             for name in result.score.__fields__.keys():
#                 if name not in vals:
#                     vals[name] = []
#                 vals[name].append(result.score.__dict__[name])

#         for k, vs in sorted(vals.items()):
#             final: float = sum(vs) / len(vs)
#             outf.write(f'{k}\t{final:.3}\t{len(vs)}\n')


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, action="append", required=True)
    oparser.add_argument("--gold", "-g", type=Path)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--txt", "-t", type=Path, required=False)
    oparser.add_argument("--max_nbest_for_txt", type=int, default=0)
    oparser.add_argument("--stat", action="store_true")
    oparser.add_argument("--csv", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    assert len(opts.input) > 0

    if opts.stat:
        stat(opts.input, opts.output)
    elif opts.csv:
        csv_output(opts.input[0], opts.output)
    else:
        assert opts.gold is not None
        assert opts.txt is not None
        evaluate_from_file(
            path_gold=opts.gold,
            path_in=opts.input[0],
            path_out=opts.output,
            path_txt_dir=opts.txt,
            nbest_idx_list=[v for v in range(-1, opts.max_nbest_for_txt)],
        )


if __name__ == "__main__":
    main()
