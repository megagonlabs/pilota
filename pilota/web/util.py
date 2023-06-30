#!/usr/bin/env python3
import os
import time
from pathlib import Path
from typing import Final, Optional

from bunkai import Bunkai
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pilota.predict import DIR_NAME_SCORER, Predictor, PredictorParameter
from pilota.schema import Request
from pilota.util import get_real_model_path
from pilota.web.schema import DEFAULT_STR, BasicInfo, Info, WebRequest, WebRequestAdminModel, WebResponse


def _delete_a_model(
    request: WebRequestAdminModel,
    name2predictor: dict[str, Predictor],
    info: Info,
) -> Optional[JSONResponse]:
    assert request.path is None
    if request.model_name is None:
        return

    if request.model_name not in name2predictor:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=f"{request.model_name} is not available",
        )
    if len(name2predictor) == 1:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content="Deletion is not permitted when the number of names is 1",
        )

    del name2predictor[request.model_name]
    info.models.remove(request.model_name)
    del info.configs[request.model_name]


def _load_a_model(
    request: WebRequestAdminModel,
    name2predictor: dict[str, Predictor],
    info: Info,
    dohalf: bool,
) -> Optional[JSONResponse]:
    assert request.path is not None
    assert request.model_name is not None
    if request.model_name == DEFAULT_STR:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=f"Name {DEFAULT_STR} is forbidden",
        )

    _p = Path(request.path)
    if not _p.exists():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=f"Path {_p} does not exist",
        )

    try:
        predictor = Predictor(
            path_model=_p,
            dohalf=dohalf,
            no_scorer=not _p.joinpath(DIR_NAME_SCORER).exists(),
        )
        name2predictor[request.model_name] = predictor
        info.models.append(request.model_name)
        info.configs[request.model_name] = predictor.config
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=f"Exception {e}",
        )
    if info.default is None:
        info.default = request.model_name


def get_app(
    path_models: list[str],
    names: list[str],
    dohalf: bool,
    static_path: Path,
):
    if len(path_models) != len(names):
        raise IndexError("Length mismatch")
    if len(set(path_models)) != len(names):
        raise KeyError("Name duplication")

    app = FastAPI(
        title="Pilota",
    )

    bki: Bunkai = Bunkai(path_model=None)
    name2predictor: dict[str, Predictor] = {}
    default_predictor_param = PredictorParameter()
    info: Info = Info.get(
        models=[],
        configs={},
        default_param=default_predictor_param,
        default_name=None,
    )
    real_path_models: list[Path] = [get_real_model_path(p) for p in path_models]
    for _p, _name in zip(real_path_models, names):
        ret: Optional[JSONResponse] = _load_a_model(
            request=WebRequestAdminModel(
                model_name=_name,
                path=str(_p),
                new_default=names[0],
                admin_key="",
            ),
            name2predictor=name2predictor,
            info=info,
            dohalf=dohalf,
        )
        if ret is not None:
            raise KeyError(ret.body)

    @app.get("/api/info")
    def get_info() -> Info:
        return info

    @app.post("/api/predict", response_model=WebResponse)
    def predict(request: WebRequest):
        if request.utterance is None and request.sentences is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content="sentences and utterance are None",
            )
        if len(name2predictor) == 0 or info.default is None:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content="No model is available")

        if request.model_name == DEFAULT_STR:
            request.model_name = info.default
        if request.model_name not in name2predictor:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=None)

        time_start_preparation: Final[float] = time.perf_counter()
        time_info: dict[str, float] = {}
        sents: list[str]
        if request.utterance is not None:
            sents = [sent for sent in bki(Predictor.normalize(request.utterance))]
        else:
            assert request.sentences is not None
            sents = [Predictor.normalize(s) for s in request.sentences]

        _req: Request = Request(
            context=request.context,
            sentences=sents,
        )
        myparam = request.param
        if myparam is None:
            myparam = default_predictor_param

        _p = name2predictor[request.model_name]
        _inputs = Predictor.get_inputs(request=_req, pc=_p.config)
        time_info["preprocess"] = time.perf_counter() - time_start_preparation

        time_start_generation: Final[float] = time.perf_counter()
        try:
            generated_ids = _p.generate_without_decode(
                sources=_inputs,
                param=myparam,
            )
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=str(e),
            )
        time_info["generate"] = time.perf_counter() - time_start_generation

        time_start_decode: Final[float] = time.perf_counter()
        scuds = _p.decode(generated_ids)
        time_info["decode"] = time.perf_counter() - time_start_decode

        scores = None
        scores_detail = None
        original_ranks = []
        for _ in range(len(sents)):
            original_ranks += [_r for _r in range(myparam.nbest)]
        if _p.scorer:
            time_start_score: Final[float] = time.perf_counter()
            scores, scores_detail = _p.scorer.calc(
                sources=_inputs,
                scuds_list=scuds,
                in_max_length=myparam.in_max_length,  # Currently use the same value to generation
            )
            if myparam.rerank:
                _p.scorer.rerank(
                    scuds_list=scuds,
                    original_ranks=original_ranks,
                    scores=scores,
                    scores_detail=scores_detail,
                    num_sentences=len(sents),
                    force_none_lowrank=myparam.force_none_lowrank,
                )
            time_info["score"] = time.perf_counter() - time_start_score

        bi: BasicInfo = BasicInfo(
            packages=info.packages,
            param=myparam,
        )

        if request.only_best:
            scuds = [scuds[x * myparam.nbest] for x in range(len(sents))]
            if scores_detail is not None:
                scores_detail = [scores_detail[x * myparam.nbest] for x in range(len(sents))]

        return WebResponse(
            scuds=scuds,
            original_ranks=None if request.only_best else original_ranks,
            scores_detail=scores_detail,
            scores=None if request.only_best else scores,
            sentences=sents,
            info=bi,
            model_name=request.model_name,
            time=time_info,
        )

    @app.post("/api/admin/model")
    def admin_model(request: WebRequestAdminModel):
        if "ADMIN_KEY" not in os.environ:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content="No ADMIN_KEY is set",
            )

        if request.admin_key != os.environ.get("ADMIN_KEY"):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content="Invalid admin key",
            )

        if (
            (request.new_default not in name2predictor)
            and (request.path is not None)
            and (request.model_name is not None)
            and (request.new_default is not None)
            and (request.new_default != request.model_name)
        ):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=f"Invalid default name: {request.new_default}",
            )

        if request.path is None:
            if request.new_default == request.model_name:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content="Default name should not be a deleted model name",
                )
            if (ret := _delete_a_model(request, name2predictor, info)) is not None:
                return ret
        elif request.model_name is not None:
            if (ret := _load_a_model(request, name2predictor, info, dohalf)) is not None:
                return ret

        # set default
        if request.new_default is None:
            pass
        elif request.new_default in name2predictor:
            info.default = request.new_default
        else:
            info.default = sorted(list(name2predictor.keys()))[0]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=f"OK. New default: {info.default}",
        )

    app.mount(
        "/",
        StaticFiles(directory=static_path, html=True),
        name="static",
    )

    return app
