#!/usr/bin/env python3

from typing import Optional

import pkg_resources
from pydantic import BaseModel, ConfigDict, validator

from pilota.predict import PredictorParameter
from pilota.schema import PilotaConfig, Request

DEFAULT_STR: str = "default"


class BasicInfo(BaseModel):
    packages: dict[str, Optional[str]]
    param: PredictorParameter


class Info(BasicInfo):
    models: list[str]
    configs: dict[str, PilotaConfig]
    default: Optional[str] = None

    @staticmethod
    def get_git_revision_short_hash() -> str:
        import subprocess

        try:
            short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            short_hash = str(short_hash, "utf-8").strip()
        except subprocess.CalledProcessError as e:
            return e.stderr
        return short_hash

    @staticmethod
    def get(
        *,
        models: list[str],
        configs: dict[str, PilotaConfig],
        default_param: PredictorParameter,
        default_name: Optional[str],
    ) -> "Info":
        _pks: dict[str, Optional[str]] = {}
        for libname in ["pytorch-lightning", "transformers"]:
            try:
                _pks[libname] = pkg_resources.get_distribution(libname).version
            except pkg_resources.DistributionNotFound:
                _pks[libname] = None
        _pks["pilota"] = Info.get_git_revision_short_hash()

        return Info(
            packages=_pks,
            models=models,
            configs=configs,
            param=default_param,
            default=default_name,
        )


class WebRequest(Request):
    model_config = ConfigDict(protected_namespaces=())  # https://github.com/pydantic/pydantic/issues/6322

    param: Optional[PredictorParameter] = None
    model_name: str = DEFAULT_STR
    only_best: bool = True


class WebResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scuds: list[list[str]]
    original_ranks: Optional[list[int]]
    scores_detail: Optional[list[dict[str, float]]]
    scores: Optional[list[float]]
    sentences: list[str]
    info: BasicInfo
    time: dict[str, float]
    model_name: str


class WebRequestAdminModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: Optional[str] = None
    path: Optional[str] = None
    new_default: Optional[str] = None
    admin_key: str

    @validator("model_name")
    def validate_name(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("must not be blank")
        return v
