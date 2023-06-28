#!/usr/bin/env python3

import os
import queue
import sys
import threading
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import Callable, Final, Iterator, Optional

import torch
from bunkai import Bunkai
from more_itertools import chunked
from pydantic import BaseModel, Field
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

import pilota.const.metachar
from pilota.const.metachar import MARK_AGENT, MARK_CONTEXT, MARK_USER, SEP, TARGET_END, TARGET_START
from pilota.schema import PilotaConfig, Request
from pilota.scorer import Scorer

logger = getLogger(__name__)
CONFIG_FILE_NAME: Final[str] = "pilota.config.json"

DIR_NAME_SCORER: Final[str] = "scorer"
DIR_NAME_SCUD: Final[str] = "scud"


class PredictorParameter(BaseModel):
    in_max_length: int = Field(default=128, ge=0)  # 0 means infinite
    out_max_length: int = Field(default=64, ge=1)
    batch_size: int = Field(default=1, ge=1)
    beam_search: int = Field(default=5, ge=1)
    nbest: int = Field(default=5, ge=1)
    repetition_penalty: float = Field(default=2.5, gt=0.0)
    length_penalty: float = 1.0
    temperature: float = 1.0

    diversity_penalty: float = 0.0
    num_beam_groups: int = Field(default=1, ge=1)
    early_stopping: bool = True
    rerank: bool = True
    force_none_lowrank: bool = False

    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    typical_p: float = Field(default=1.0, ge=0.0, le=1.0)


WrapDecodeReturnType = tuple[
    list[list[str]],
    list[str],
    list[str],
    Optional[list[float]],
    Optional[list[dict[str, float]]],
    list[bool],
    int,
    bool,
    list[int],
    bool,
]

Results = list[str]
NBestResults = list[list[Results]]  # NBestResults[nbest][sent_idx]


class ResultForSentence(BaseModel):
    scuds_nbest: list[Results]
    original_ranks: list[int]
    scores: Optional[list[float]]
    scores_detail: Optional[list[dict[str, float]]]
    sentence: str


class Predictor:
    path_model: Path
    model: AutoModelForSeq2SeqLM
    device: str
    tokenizer: PreTrainedTokenizer
    raw_in: bool
    scorer: Optional[Scorer] = None

    def __init__(
        self,
        *,
        path_model: Path,
        device: str = "cpu",
        bki: Optional[Bunkai] = None,
        queue_max_size: int = 4,
        num_decoder: Optional[int] = None,
        dohalf: bool = False,
        raw_in: bool = False,
        no_scorer: bool = False,
    ):
        self.path_model = path_model
        self.device: str = device
        path_model_scud: Path = path_model.joinpath(DIR_NAME_SCUD)
        self.config = PilotaConfig.parse_file(path_model_scud.joinpath(CONFIG_FILE_NAME))
        self.threads = []

        _m = AutoModelForSeq2SeqLM.from_pretrained(path_model_scud)
        self.model = _m.to(device)
        self.model.eval()
        if dohalf:
            self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(path_model_scud, use_fast=False)

        if num_decoder is None:
            num_decoder = os.cpu_count()
            if num_decoder is None:
                num_decoder = 1

        self.num_decoder = num_decoder
        self.raw_in = raw_in

        if not no_scorer:
            path_scorer = self.path_model.joinpath(DIR_NAME_SCORER)
            if path_scorer.exists():
                self.scorer = Scorer(path_scorer)
            else:
                raise FileNotFoundError(f"{path_scorer} does not exist")

        # ----
        func_params = [
            (
                wrap_gen,
                (bki, self),
            ),
            (
                wrap_decode,
                (self,),
            ),
        ]
        if self.scorer:
            func_params.append(
                (
                    wrap_rerank,
                    (self,),
                )
            )

        wrap_funcs = list(map(lambda func_param: wrap_func_for_mt(func_param[0], func_param[1]), func_params))
        self.queues_input = [queue.Queue(maxsize=queue_max_size) for _ in range(len(wrap_funcs) + 1)]
        for idx, wrap_func in enumerate(wrap_funcs):
            t = threading.Thread(target=wrap_func, args=(self.queues_input[idx], (self.queues_input[idx + 1])))
            self.threads.append(t)

        for t in self.threads:
            t.start()
        # ----

    def queue_put(self, arg: Optional[tuple[Path, PredictorParameter]]):
        self.queues_input[0].put(arg)  # To the first one of the pipeline

    def queue_get(self) -> Optional[WrapDecodeReturnType]:
        return self.queues_input[-1].get()  # From the last one of the pipeline

    def queue_iter(self) -> Iterator[list[ResultForSentence]]:
        tmp_pred_store: Optional[NBestResults] = None
        tmp_sents = []
        tmp_scores = []
        tmp_scores_detail = []
        while True:
            r = self.queue_get()
            if r is None:  # normal termination
                return
            elif not isinstance(r, tuple):  # pass-through exception
                sys.exit(1)
            logger.debug("Get one from queue")

            preds, sents, _, scores, scores_detail, flags, _, _, original_ranks, _ = r
            nbest: int = int(len(preds) / len(flags))

            pred_candidates: list[list[str]]
            for idx, (pred_candidates, sent, flag) in enumerate(zip(chunked(preds, nbest), sents, flags)):
                logger.debug(f" loop in #{idx}")
                myscores = None
                myscores_detail = None
                if scores is not None:
                    myscores = scores[idx * nbest : (idx + 1) * nbest]
                    assert scores_detail is not None
                    myscores_detail = scores_detail[idx * nbest : (idx + 1) * nbest]
                my_original_ranks = original_ranks[idx * nbest : (idx + 1) * nbest]

                if tmp_pred_store is None:
                    tmp_pred_store = [[] for _ in range(nbest)]
                for j, p in enumerate(pred_candidates):
                    tmp_pred_store[j].append(p)

                tmp_sents.append(sent)
                tmp_scores.append(myscores)
                tmp_scores_detail.append(myscores_detail)
                if flag:
                    ret = []
                    for j, (sent, scores_for_sent, scores_detail_for_sent) in enumerate(
                        zip(
                            tmp_sents,
                            tmp_scores,
                            tmp_scores_detail,
                        )
                    ):
                        ret.append(
                            ResultForSentence(
                                scuds_nbest=[tmp_pred_store[k][j] for k in range(nbest)],
                                original_ranks=my_original_ranks,
                                scores=scores_for_sent,
                                scores_detail=scores_detail_for_sent,
                                sentence=sent,
                            )
                        )
                    yield ret
                    tmp_pred_store = None
                    tmp_sents = []
                    tmp_scores = []

    def __del__(self):
        for t in self.threads:
            t.join()

    @staticmethod
    def normalize(query: str) -> str:
        return (
            unicodedata.normalize("NFKC", query)
            .replace("\t", " ")
            .replace("\n", pilota.const.metachar.LINE_BREAK)
            .strip()
        )

    @staticmethod
    def decode_to_scuds(q: str) -> list[str]:
        q = q.strip().replace(pilota.const.metachar.NONE, "")
        if len(q) == 0:
            return []
        r = [v.strip() for v in q.split(pilota.const.metachar.SEP)]
        return list(filter(lambda v: len(v) != 0, r))

    @staticmethod
    def get_context_str_for_dialog(*, request: Request, pc: PilotaConfig) -> str:
        if pc.size_context == 0:
            return ""
        assert request.context is not None

        rets: list[str] = []
        _from = max(0, len(request.context) - pc.size_context)
        for idx in reversed(range(_from, len(request.context))):
            if request.context[idx].name == "agent":
                rets.append(MARK_AGENT)
            else:
                rets.append(MARK_USER)
            rets.append(request.context[idx].text)
        return "".join(rets)

    @staticmethod
    def get_inputs(
        *,
        request: Request,
        pc: PilotaConfig,
        monologue_training: bool = False,
    ) -> list[str]:
        assert request.sentences is not None
        ret = []

        if len(pc.acceptable_names) == 1:  # monologue
            if monologue_training:
                assert len(request.sentences) == 1
                assert request.context is not None
                if pc.size_context == 0:
                    ret.append(request.sentences[0])
                else:
                    ret.append(
                        request.sentences[0]
                        + MARK_CONTEXT
                        + SEP.join([v.text for v in reversed(request.context[-pc.size_context :])])
                    )
            else:
                assert request.context is None
                for idx, sent in enumerate(request.sentences):
                    if pc.size_context == 0:
                        ret.append(sent)
                    else:
                        _from: int = max(0, idx - pc.size_context)
                        ret.append(sent + MARK_CONTEXT + SEP.join([v for v in reversed(request.sentences[_from:idx])]))
        else:
            assert request.context is not None
            _ctx: str = Predictor.get_context_str_for_dialog(
                request=request,
                pc=pc,
            )
            for idx, sent in enumerate(request.sentences):
                out = request.sentences[:]
                out[idx] = TARGET_START + sent + TARGET_END
                ret.append("".join(out) + MARK_CONTEXT + _ctx)
        return ret

    @staticmethod
    def generate(
        model,
        input_ids,
        attention_mask,
        param: PredictorParameter,
    ) -> torch.LongTensor:
        # https://huggingface.co/docs/transformers/main_classes/text_generation
        return model.generate(
            input_ids=input_ids,  # type: ignore
            attention_mask=attention_mask,  # type : ignore
            max_length=param.out_max_length,
            #                 truncation=True,
            num_beams=param.beam_search,
            num_return_sequences=param.nbest,
            repetition_penalty=param.repetition_penalty,
            length_penalty=param.length_penalty,
            temperature=param.temperature,
            diversity_penalty=param.diversity_penalty,
            num_beam_groups=param.num_beam_groups,
            early_stopping=param.early_stopping,
            top_k=param.top_k,
            top_p=param.top_p,
            typical_p=param.typical_p,
        )

    def generate_without_decode(
        self,
        *,
        sources: list[str],
        param: PredictorParameter,
    ) -> torch.Tensor:
        source = self.tokenizer.batch_encode_plus(
            sources,
            padding="longest",
            truncation=True,
            max_length=param.in_max_length if param.in_max_length > 0 else None,
            return_tensors="pt",
        )
        #     print(tokenizer.convert_ids_to_tokens(source['input_ids'][0].tolist()))

        source.to(self.device)

        with torch.no_grad():
            generated_ids = Predictor.generate(
                self.model,
                source["input_ids"],
                source["attention_mask"],
                param,
            )
        return generated_ids.to("cpu")

    def decode(self, generated_ids: torch.Tensor) -> list[list[str]]:
        return para_decode(self.num_decoder, generated_ids, self.tokenizer)


def para_decode(num: int, generated_ids: torch.Tensor, tokenizer) -> list[list[str]]:
    pool = ProcessPoolExecutor(max_workers=num)
    with pool as executor:
        tasks = [executor.submit(_mydecode, i, g, tokenizer) for i, g in enumerate(generated_ids)]

    tmp = [future.result() for future in as_completed(tasks)]
    prediction = [r for _, r in sorted(tmp, key=lambda x: x[0])]
    return prediction


WrapGenReturnType = Optional[tuple[torch.Tensor, list[str], list[str], list[bool], int, bool, bool]]
WrapGenArgType = tuple[Path, PredictorParameter]
WrapGenParamType = tuple[Bunkai, Predictor]


def wrap_gen(args: WrapGenArgType, params: WrapGenParamType) -> Iterator[WrapGenReturnType]:
    path_in, param = args
    bki, predictor = params

    def _iter_input() -> Iterator[tuple[str, str, bool]]:
        with path_in.open() as inf:
            for line in inf:
                if predictor.raw_in:
                    yield line[:-1], "", True
                    continue
                req: Request = Request.parse_raw(line)
                sents: list[str]
                if req.sentences is not None:
                    sents = req.sentences
                else:
                    assert req.utterance is not None
                    sents = [s for s in bki(req.utterance)]
                    req = Request(
                        context=req.context,
                        sentences=sents,
                    )
                inps = Predictor.get_inputs(
                    request=req,
                    pc=predictor.config,
                )
                for idx, inp in enumerate(inps):
                    yield inp, sents[idx], idx == len(inps) - 1

    for pairs in chunked(_iter_input(), param.batch_size):
        sources: list[str] = [p[0] for p in pairs]
        sents: list[str] = [p[1] for p in pairs]
        flags: list[bool] = [p[2] for p in pairs]
        logger.debug(f"Generate for {len(sources)} sources")
        time_start: Final[float] = time.perf_counter()
        generated_ids = predictor.generate_without_decode(
            sources=sources,
            param=param,
        )
        logger.debug(f"Generation finished ({time.perf_counter()-time_start:0.3f} sec)")
        yield (
            generated_ids,
            sents,
            sources,
            flags,
            param.in_max_length,
            param.force_none_lowrank,
            param.rerank,
        )
    yield None


WrapDecodeParamType = tuple[Predictor]


def wrap_decode(args: WrapGenReturnType, params: WrapDecodeParamType) -> Iterator[WrapDecodeReturnType]:
    assert args is not None

    generated_ids, sents, sources, flags, in_max_length, force_none_lowrank, do_rerank = args
    predictor = params[0]
    results = predictor.decode(generated_ids)

    nbest: int = int(len(results) / len(flags))
    original_ranks = []
    for _ in range(len(sents)):
        original_ranks += [_r for _r in range(nbest)]

    yield results, sents, sources, None, None, flags, in_max_length, force_none_lowrank, original_ranks, do_rerank,


def wrap_rerank(args: WrapDecodeReturnType, params: WrapDecodeParamType) -> Iterator[WrapDecodeReturnType]:
    assert args is not None
    scuds, sents, sources, _, _, flags, in_max_length, force_none_lowrank, original_ranks, do_rerank = args
    predictor = params[0]
    assert predictor.scorer is not None
    logger.debug(f"Score calculation for {len(sources)} sources")
    time_start: Final[float] = time.perf_counter()
    scores, scores_detail = predictor.scorer.calc(
        sources=sources,
        scuds_list=scuds,
        in_max_length=in_max_length,
    )
    if do_rerank:
        predictor.scorer.rerank(
            scuds_list=scuds,
            original_ranks=original_ranks,
            scores=scores,
            scores_detail=scores_detail,
            num_sentences=len(sents),
            force_none_lowrank=force_none_lowrank,
        )
    logger.debug(f"Score calculation finished ({time.perf_counter()-time_start:0.3f} sec)")
    yield (
        scuds,
        sents,
        sources,
        scores,
        scores_detail,
        flags,
        in_max_length,
        force_none_lowrank,
        original_ranks,
        do_rerank,
    )


def wrap_func_for_mt(func, params) -> Callable:
    def wrap_func(queue_input, queue_output):
        while True:
            args = queue_input.get()
            if args is None:
                queue_output.put(None)
                break
            elif not isinstance(args, tuple):  # pass-through exception
                queue_output.put(args)
                break

            try:
                for result in func(args, params):
                    queue_output.put(result)
            except Exception as e:
                import traceback

                traceback.print_exc()
                queue_output.put(e)

        queue_input.task_done()
        return

    return wrap_func


def _mydecode(i, g, tokenizer) -> tuple[int, list[str]]:
    return i, Predictor.decode_to_scuds(
        tokenizer.decode(
            g,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
