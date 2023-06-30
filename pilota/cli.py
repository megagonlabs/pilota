#!/usr/bin/env python3

import argparse
import os
import sys
from logging import basicConfig, getLogger
from pathlib import Path
from typing import Optional

import torch
from bunkai import Bunkai

from pilota.predict import Predictor, PredictorParameter, ResultForSentence
from pilota.util import get_real_model_path

logger = getLogger(__name__)


def operation(
    *,
    path_model_str: str,
    path_in: Path,
    path_out: Path,
    batch_size: int,
    inlen: int,
    outlen: int,
    raw_in: bool,
    beam_search: int,
    nbest: int,
    repetition_penalty: float,
    num_decoder: Optional[int],
    dohalf: bool,
    print_param: bool,
    queue_max_size: int,
    no_scorer: bool,
) -> None:
    if nbest > beam_search:
        sys.stderr.write(f'Changed "beam_search" {beam_search} to {nbest}\n')
        beam_search = nbest

    param = PredictorParameter(
        in_max_length=inlen,
        out_max_length=outlen,
        batch_size=batch_size,
        beam_search=beam_search,
        nbest=nbest,
        repetition_penalty=repetition_penalty,
    )
    if print_param:
        with path_out.open("w") as outf:
            outf.write(param.json(indent=4))
            outf.write("\n")
        return

    bki: Optional[Bunkai] = None
    if not raw_in:
        bki = Bunkai(path_model=None)

    path_model: Path = get_real_model_path(path_model_str)
    logger.info(f"Loading {path_model}")
    predictor = Predictor(
        path_model=path_model,
        num_decoder=num_decoder,
        device="cuda" if torch.cuda.is_available() else "cpu",
        bki=bki,
        dohalf=dohalf,
        raw_in=raw_in,
        no_scorer=no_scorer,
        queue_max_size=queue_max_size,
    )
    logger.info("Predictor loaded")

    with path_out.open("w") as outf:
        predictor.queue_put((path_in, param))
        predictor.queue_put(None)
        ret_list: list[ResultForSentence]
        for ret_list in predictor.queue_iter():
            outf.write("[")
            for i, ret in enumerate(ret_list):
                if i > 0:
                    outf.write(", ")
                outf.write(ret.json(ensure_ascii=False))
            outf.write("]\n")
    logger.info("Bye")


def get_opts() -> argparse.ArgumentParser:
    dpp = PredictorParameter(out_max_length=1)

    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--model", "-m", required=True)
    oparser.add_argument("--bs", "--batch_size", type=int, default=dpp.batch_size)
    oparser.add_argument("--il", "--inlen", type=int, default=dpp.in_max_length, help="0 means unlimited")
    oparser.add_argument("--ol", "--outlen", type=int, required=True)
    oparser.add_argument("--raw_in", action="store_true")
    oparser.add_argument("--dohalf", action="store_true")
    oparser.add_argument("--print_param", action="store_true")
    oparser.add_argument("--no_scorer", action="store_true")

    oparser.add_argument("--decoder", "--dn", type=int)
    oparser.add_argument("--queue", type=int, default=4, help="Max size of queue")

    oparser.add_argument("--beam", type=int, default=dpp.beam_search)
    oparser.add_argument("--nbest", type=int, default=dpp.nbest)
    oparser.add_argument("--repp", type=float, default=dpp.repetition_penalty)
    oparser.add_argument("--lenp", type=float, default=dpp.length_penalty)
    return oparser


def main() -> None:
    basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING"),
        format="[%(asctime)s] %(module)s.%(funcName)s:%(lineno)d %(levelname)s -> %(message)s",
    )
    opts = get_opts().parse_args()
    operation(
        path_model_str=opts.model,
        path_in=opts.input,
        path_out=opts.output,
        batch_size=opts.bs,
        inlen=opts.il,
        outlen=opts.ol,
        raw_in=opts.raw_in,
        beam_search=opts.beam,
        nbest=opts.nbest,
        repetition_penalty=opts.repp,
        num_decoder=opts.decoder,
        dohalf=opts.dohalf,
        print_param=opts.print_param,
        queue_max_size=opts.queue,
        no_scorer=opts.no_scorer,
    )


if __name__ == "__main__":
    main()
