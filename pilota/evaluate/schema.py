#!/usr/bin/env python3

from typing import Optional

from pydantic import BaseModel


class RougeScore(BaseModel):
    rouge_1: float
    rouge_2: float
    rouge_l: float


class ScudsResult(BaseModel):
    text: str
    scuds_tokenized: list[list[str]]
    score: Optional[dict[str, float]]

    rouge_score: RougeScore


class Result(BaseModel):
    sid: str

    input: str
    source_text: str
    source_text_tokenized: list[str]

    target_gold: str
    target_golds_tokenized: list[list[str]]

    target_predicted: dict[int, ScudsResult]
