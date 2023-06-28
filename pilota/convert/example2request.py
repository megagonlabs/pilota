#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional, get_args

from asdc.schema.example import Example

from pilota.const.metachar import NONE, SEP
from pilota.predict import Predictor
from pilota.schema import NAME, PilotaConfig, Request, Utterance


def example2scorer_instance(
    *,
    ex: Example,
    pc: PilotaConfig,
    return_req: bool,
) -> str:
    req = Request(
        context=[Utterance.parse_obj(v) for v in ex.context],
        sentences=[s for s in ex.sources],
        meta={
            "sid": ex.sid,
        },
    )
    if return_req:
        return req.json(ensure_ascii=False)

    monologue_training = False
    if len(pc.acceptable_names) == 1:  # monologue
        monologue_training = True
        assert ex.sid.sentence_num == 0

    _src: str = Predictor.get_inputs(
        request=req,
        pc=pc,
        monologue_training=monologue_training,
    )[ex.sid.sentence_num]

    _tgt: str = NONE
    if len(ex.targets) > 0:
        _tgt = SEP.join(ex.targets)
    assert len(_tgt) > 0

    _src = Predictor.normalize(_src)
    _tgt = Predictor.normalize(_tgt)
    assert "\t" not in _src
    assert "\t" not in _tgt

    return "\t".join(
        [
            Predictor.normalize(ex.sid.id),
            _src,
            _tgt,
        ]
    )


def convert(
    *,
    path_in: Path,
    path_out: Path,
    pc: PilotaConfig,
    as_tsv: bool,
):
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            ex: Example = Example.parse_raw(line)
            assert ex.correct is True

            _out = example2scorer_instance(
                ex=ex,
                pc=pc,
                return_req=not as_tsv,
            )
            outf.write(_out)
            outf.write("\n")


def operation(
    *,
    path_in: Path,
    path_out: Path,
    path_out_config: Optional[Path],
    context: int,
    acceptable_names: list[str],
    as_tsv: bool,
) -> None:
    assert len(acceptable_names) > 0
    pc = PilotaConfig(
        size_context=context,
        acceptable_names=acceptable_names,
    )

    if path_out_config:
        path_out_config.parent.parent.mkdir(parents=True, exist_ok=True)
        with path_out_config.open("w") as outfs:
            outfs.write(f"{pc.json(indent=4)}\n")

    if path_in.is_dir():
        path_out.mkdir(parents=True, exist_ok=True)
        fnames = ["train", "test", "dev"]
        for fname in fnames:
            convert(
                path_in=path_in.joinpath(f"{fname}.jsonl"),
                path_out=path_out.joinpath(f"{fname}.tsv"),
                pc=pc,
                as_tsv=as_tsv,
            )
    else:
        convert(
            path_in=path_in,
            path_out=path_out,
            pc=pc,
            as_tsv=as_tsv,
        )


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, required=True)
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    oparser.add_argument("--output_config", "-c", type=Path, required=False)

    oparser.add_argument("--context", type=int, required=True)
    oparser.add_argument("--name", action="append", required=True, choices=get_args(NAME))
    oparser.add_argument("--tsv", action="store_true")
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    operation(
        path_in=opts.input,
        path_out=opts.output,
        path_out_config=opts.output_config,
        context=opts.context,
        acceptable_names=opts.name,
        as_tsv=opts.tsv,
    )


if __name__ == "__main__":
    main()
