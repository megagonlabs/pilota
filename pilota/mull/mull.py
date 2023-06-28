#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Final, Optional, Union

import numpy as np
import torch
import torch.utils.data
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


def get_tokenizer(
    *,
    model_name: str,
    is_fast: bool,
    path_model: Optional[Path] = None,
):
    no_position_ids: bool = False
    my_path = model_name
    if path_model is not None:
        my_path = path_model
    if model_name == "rinna/japanese-roberta-base":
        tokenizer = T5Tokenizer.from_pretrained(
            my_path,
            is_fast=is_fast,
        )
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    elif model_name == "line-corporation/line-distilbert-base-japanese":
        no_position_ids = True
        tokenizer = AutoTokenizer.from_pretrained(
            my_path,
            revision="bcbdf6df31d2ef58bb6288f7aa4906fc688969a5",
            trust_remote_code=True,
            is_fast=is_fast,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            my_path,
            is_fast=is_fast,
        )
    return tokenizer, no_position_ids


def encode(
    *,
    tokenizer: PreTrainedTokenizer,
    inputs: Union[list[str], list[list[str]]],
    in_max_length: int,
    model_name: str,
    for_train: bool,
) -> BatchEncoding:
    source: BatchEncoding = tokenizer.batch_encode_plus(
        inputs,
        padding="max_length" if for_train else "longest",
        truncation=True,
        max_length=in_max_length,
        return_tensors="pt",
    )

    if model_name == "rinna/japanese-roberta-base":
        # https://github.com/rinnakk/japanese-pretrained-models/issues/3
        position_ids = [list(range(0, source["input_ids"].size(1)))] * source["input_ids"].size(0)  # type:ignore
        position_id_tensor = torch.LongTensor(position_ids)
        source["position_ids"] = position_id_tensor
    return source


class Mull:
    tokenizer: PreTrainedTokenizer
    no_position_ids: bool
    labels: list[str]
    model: AutoModelForSequenceClassification
    device: str
    original_name: str
    singlebinary: bool
    in_max_length: int

    def __init__(self):
        pass

    @staticmethod
    def from_pretrained(
        path_model: Path,
        is_fast: bool = True,
    ) -> "Mull":
        if not path_model.is_dir():
            raise NotADirectoryError(f"{path_model} is invalid")

        self = Mull()

        config = AutoConfig.from_pretrained(path_model)
        self.labels = config.my_labels
        self.singlebinary = config.singlebinary
        with path_model.joinpath("config.json").open() as cf:
            self.original_name = json.load(cf)["_name_or_path"]

        self.tokenizer, self.no_position_ids = get_tokenizer(
            model_name=self.original_name,
            path_model=path_model,
            is_fast=is_fast,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(path_model).to(self.device)
        self.model.eval()

        self.in_max_length: Final[int] = self.model.config.max_position_embeddings - 2  # type: ignore

        return self

    @torch.no_grad()
    def predict(
        self,
        *,
        inputs: Union[list[str], list[list[str]]],
        in_max_length: Optional[int] = None,
    ) -> np.ndarray:
        source = encode(
            tokenizer=self.tokenizer,
            inputs=inputs,
            in_max_length=self.in_max_length if in_max_length is None else min(in_max_length, self.in_max_length),
            model_name=self.original_name,
            for_train=False,
        )
        source.to(self.device)

        kv = {}
        if not self.no_position_ids and "position_ids" in source:
            kv["position_ids"] = source["position_ids"]
        outputs = self.model(  # type: ignore
            input_ids=source["input_ids"],
            attention_mask=source["attention_mask"],
            **kv,
        )
        predictions = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        if self.singlebinary:
            predictions = np.delete(predictions, 0, axis=1)
        return predictions
