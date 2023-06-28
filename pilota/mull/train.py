#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.tokenization_utils_base import BatchEncoding

from pilota.mull.mull import encode, get_tokenizer

DEFAULT_SEP: str = "<pair>"


def labelfile_to_labels(label: Path) -> list[str]:
    labels: list[str] = []
    with label.open() as lf:
        for line in lf:
            labels.append(line.strip())
    assert len(labels) == len(set(labels))  # no duplication
    return labels


class TsvDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        file_path: Path,
        input_max_len: int,
        label: Path,
        base: str,
        singlebinary: bool,
        separator: Optional[str],
    ):
        self.file_path = file_path

        self.input_max_len = input_max_len
        self.labels: list[str] = labelfile_to_labels(label)
        self.inputs: list[BatchEncoding] = []
        self.targets: list[list[Union[int, float]]] = []
        self.base = base
        self.singlebinary = singlebinary
        self.separator = separator

        self._build(tokenizer)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"][0]  # type: ignore
        source_mask = self.inputs[index]["attention_mask"][0]  # type: ignore
        targets = self.targets[index]

        t_arg = {}
        if self.singlebinary:
            t_arg["dtype"] = torch.int64
        tmp = {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "targets": torch.tensor(targets, **t_arg),
        }
        if "position_ids" in self.inputs[index]:
            tmp["position_ids"] = self.inputs[index]["position_ids"][0]  # type: ignore
        return tmp

    def _get_mytarget(self, val: str) -> list[Union[int, float]]:
        rets: list[Union[int, float]]
        if self.singlebinary:
            rets = [0]
        else:
            rets = [0.0] * len(self.labels)
        if len(val) == 0:
            return rets

        if val.startswith("["):
            mylabels = json.loads(val)
        else:
            mylabels = val.split(",")

        for lb in mylabels:
            if self.singlebinary:
                rets[self.labels.index(lb)] = 1
                continue
            _value: float = 1.0
            tmp = lb.split("@@")
            if len(tmp) > 1:
                lb = tmp[0]
                _value = float(tmp[1])
            assert lb in self.labels
            rets[self.labels.index(lb)] = _value
        return rets

    def _build(self, tokenizer):
        with self.file_path.open() as f:
            for line in tqdm(f):
                items = line[:-1].split("\t")
                assert len(items) <= 3
                assert len(items[0]) > 0

                if len(items) == 2:
                    text = items[0]
                    mytarget = self._get_mytarget(items[1])
                else:
                    text = items[1]
                    assert len(text) > 0
                    mytarget = self._get_mytarget(items[2])
                assert len(text) > 0

                tin: Union[str, list[str]]
                if self.separator is None:
                    tin = text
                else:
                    tin = text.split(self.separator, maxsplit=1)
                    assert len(tin) == 2

                tokenized_inputs = encode(
                    tokenizer=tokenizer,
                    inputs=[tin],  # type: ignore
                    in_max_length=self.input_max_len,
                    model_name=self.base,
                    for_train=True,
                )
                self.inputs.append(tokenized_inputs)
                self.targets.append(mytarget)

        print(f"Size: {len(self.targets)}")


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base: str,
        il: int,
        bs: int,
        bs_dev: int,
        label: Path,
        train: Path,
        separator: Optional[str] = None,
        singlebinary: bool = False,
        dev: Optional[Path] = None,
        slow: bool = False,
        pair: bool = False,
    ):
        super().__init__()

        self.base = base
        self.slow = slow
        self.dir_train = train
        self.dir_dev = dev
        self.label = label
        self.il = il
        self.singlebinary = singlebinary
        self.separator = separator
        self.pair = pair
        self.bs = bs
        self.bs_dev = bs_dev

    def setup(
        self,
        stage: str,
    ):
        if stage == "test":
            raise NotImplementedError

        self.tokenizer, _ = get_tokenizer(
            model_name=self.base,
            is_fast=not self.slow,
        )

        self.ds_train = TsvDataset(
            tokenizer=self.tokenizer,
            file_path=self.dir_train,
            input_max_len=self.il,
            label=self.label,
            base=self.base,
            singlebinary=self.singlebinary,
            separator=self.separator if self.pair else None,
        )

        self.ds_dev = None
        if self.dir_dev:
            self.ds_dev = TsvDataset(
                tokenizer=self.tokenizer,
                file_path=self.dir_dev,
                input_max_len=self.il,
                label=self.label,
                base=self.base,
                singlebinary=self.singlebinary,
                separator=self.separator if self.pair else None,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.bs,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        if self.ds_dev is None:
            return None
        return DataLoader(
            self.ds_dev,
            batch_size=self.bs_dev,
            num_workers=4,
        )

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        assert stage is not None
        pass


class FineTuner(pl.LightningModule):
    def __init__(
        self,
        *,
        base: str,
        label: Path,
        no_position_ids: bool = False,
        singlebinary: bool = False,
    ):
        super().__init__()
        self.no_position_ids = no_position_ids

        self.labels: list[str] = labelfile_to_labels(label)

        problem_type: str = "multi_label_classification"
        carg = {}
        if singlebinary:
            problem_type = "single_label_classification"
        else:
            label_map: dict[int, str] = {i: label for i, label in enumerate(self.labels)}
            carg["id2label"] = label_map
            carg["label2id"] = {label: i for i, label in label_map.items()}
        carg["my_labels"] = self.labels

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=base,
            problem_type=problem_type,
            **carg,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base,
            config=config,
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(  # type: ignore
            **kwargs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        kv = {}
        if not self.no_position_ids and "position_ids" in batch:
            kv["position_ids"] = batch["position_ids"]
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["targets"],
            **kv,
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"test_loss": loss}


class MyLtCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults(
            {
                "seed_everything": 42,
                "trainer.logger": "TensorBoardLogger",
            }
        )

        parser.add_argument(
            "-o", "--output", type=Path, required=True, help="path to the output directory for storing the model"
        )
        parser.add_argument(
            "--early_stop_patience",
            type=int,
            default=3,
        )

        parser.link_arguments("data.singlebinary", "model.singlebinary", apply_on="instantiate")
        parser.link_arguments("data.base", "model.base", apply_on="instantiate")
        parser.link_arguments("data.label", "model.label", apply_on="instantiate")

    def before_fit(self):
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        self.my_checkpoint_callback = ModelCheckpoint(
            monitor="loss" if self.config.fit.data.dev is None else "val_loss",
            filename="model-{epoch:02d}-{step:04d}",
            verbose=True,
            save_top_k=1,
            dirpath=self.config.fit.output.joinpath("checkpoints"),
        )
        self.trainer.callbacks.append(self.my_checkpoint_callback)  # type: ignore
        if self.config.fit.early_stop_patience > 0:
            self.trainer.callbacks.append(  # type: ignore
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=self.config.fit.early_stop_patience,
                )
            )

    def after_fit(self):
        tokenizer, no_position_ids = get_tokenizer(
            model_name=self.config.fit.data.base,
            is_fast=not self.config.fit.data.slow,
        )
        tokenizer.save_pretrained(self.config.fit.output)

        labels: list[str] = labelfile_to_labels(self.config.fit.data.label)
        tuner2 = FineTuner.load_from_checkpoint(
            self.my_checkpoint_callback.best_model_path,
            base=self.config.fit.data.base,
            label=self.config.fit.data.label,
            no_position_ids=no_position_ids,
            singlebinary=self.config.fit.data.singlebinary,
        )
        tuner2.model.save_pretrained(self.config.fit.output)  # type:ignore
        with self.config.fit.output.joinpath("config.json").open() as inf:
            d = json.load(inf)
            d["my_labels"] = labels
            if self.config.fit.data.base == "rinna/japanese-roberta-base":
                d["tokenizer_class"] = "T5Tokenizer"
            d["singlebinary"] = self.config.fit.data.singlebinary
        with self.config.fit.output.joinpath("config.json").open("w") as wf:
            json.dump(d, wf, indent=4)


def main():
    _ = MyLtCLI(FineTuner, MyDataModule)


if __name__ == "__main__":
    main()
