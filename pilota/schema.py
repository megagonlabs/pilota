#!/usr/bin/env python3

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, root_validator

NAME = Literal["user", "agent"]


class Utterance(BaseModel):
    name: NAME
    text: str

    class Config:
        """Config of Utterance"""

        allow_population_by_field_name = True
        alias_generator = lambda n: {"name": "speaker"}.get(n, n)  # noqa: E731


class Request(BaseModel):
    context: Optional[list[Utterance]]
    utterance: Optional[str] = None
    sentences: Optional[list[str]] = None
    meta: dict[str, Any] = {}

    @root_validator
    def validate_source(cls, values):
        if values["utterance"] is None:
            if values["sentences"] is None:
                raise ValueError("One of `utterance` or `sentences` must be None and the other must have a value")
        else:
            if values["sentences"] is not None:
                raise ValueError("One of `utterance` or `sentences` must be None and the other must have a value")
        return values


class PilotaConfig(BaseModel):
    size_context: int = Field(default=9999, ge=0)
    acceptable_names: list[str]


class ScorerConfig(BaseModel):
    context_separator: str


Results = list[str]
NBestResult = list[Results]
NBestResults = list[NBestResult]  # NBestResults[nbest][sent_idx]


class ResultForSentence(BaseModel):
    scuds_nbest: NBestResult
    original_ranks: list[int]
    scores: Optional[list[float]]
    scores_detail: Optional[list[dict[str, float]]]
    sentence: str
