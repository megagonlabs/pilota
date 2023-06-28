#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.multiprocessing
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils_base import BatchEncoding

from pilota.const.metachar import ADDITIONAL_TOKENS


def get_tokenizer(path_model: str, add_tokens: list[str] = []) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(
        path_model,
        use_fast=False,
    )
    tokenizer.add_tokens(ADDITIONAL_TOKENS)
    if len(add_tokens) > 0:
        tokenizer.add_tokens(add_tokens)
    return tokenizer


def get_add_tokens(path_add_token: Optional[Path]):
    add_tokens = []
    if path_add_token:
        with path_add_token.open() as inf:
            for line in inf:
                w: str = line.strip()
                assert "\t" not in w
                add_tokens.append(w)
    return add_tokens


class TsvDataset(Dataset):
    input_max_len: int
    target_max_len: int
    inputs: list[BatchEncoding]
    targets: list[BatchEncoding]

    ids: list[str]

    def __init__(
        self,
        *,
        tokenizer: T5Tokenizer,
        file_path_list: list[Path],
        input_max_len: int,
        target_max_len: int,
    ):
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.inputs = []
        self.targets = []
        self.targets_text = []
        self.ids = []

        self._build(
            tokenizer=tokenizer,
            file_path_list=file_path_list,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        source_ids = self.inputs[index]["input_ids"].squeeze()  # type: ignore
        target_ids = self.targets[index]["input_ids"].squeeze()  # type: ignore

        source_mask = self.inputs[index]["attention_mask"].squeeze()  # type: ignore
        target_mask = self.targets[index]["attention_mask"].squeeze()  # type: ignore

        ret = {
            "id": self.ids[index],
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }
        return ret

    def _build(
        self,
        *,
        tokenizer: T5Tokenizer,
        file_path_list: list[Path],
    ):
        for file_path in file_path_list:
            with file_path.open() as f:
                for line in tqdm(f):
                    items = line.strip().split("\t")
                    assert len(items) == 3
                    myid: str = items[0]
                    _input: str = items[1]
                    _target: str = items[2]

                    assert len(myid) > 0
                    assert len(_input) > 0
                    assert len(_target) > 0

                    tokenized_inputs = tokenizer.batch_encode_plus(
                        [_input],
                        max_length=self.input_max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    tokenized_targets = tokenizer.batch_encode_plus(
                        [_target],
                        max_length=self.target_max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )

                    self.ids.append(myid)
                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base: str,
        il: int,
        ol: int,
        bs: int,
        bs_dev: int,
        train: list[Path],
        dev: list[Path],
        add_token: Optional[Path] = None,
    ):
        super().__init__()

        self.base = base
        self.dir_train = train
        self.dir_dev = dev
        self.il = il
        self.ol = ol
        self.bs = bs
        self.bs_dev = bs_dev
        self.path_add_token = add_token

    def setup(
        self,
        stage: str,
    ):
        if stage == "test":
            raise NotImplementedError

        # Load
        add_tokens = get_add_tokens(self.path_add_token)
        self.tokenizer = get_tokenizer(self.base, add_tokens)
        self.ds_train: TsvDataset = TsvDataset(
            tokenizer=self.tokenizer,
            file_path_list=self.dir_train,
            input_max_len=self.il,
            target_max_len=self.ol,
        )
        self.ds_dev: TsvDataset = TsvDataset(
            tokenizer=self.tokenizer,
            file_path_list=self.dir_dev,
            input_max_len=self.il,
            target_max_len=self.ol,
        )

        all_ids: list[str] = self.ds_train.ids + self.ds_dev.ids
        assert len(all_ids) == len(set(all_ids)), "Found not unique ID"

        print("train_dataset: ", len(self.ds_train))
        print("dev_dataset: ", len(self.ds_dev))

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


class T5FineTuner(pl.LightningModule):
    model_conv: T5ForConditionalGeneration

    def __init__(
        self,
        *,
        base: str,
        add_token: Optional[Path] = None,
        parallel: Optional[int] = None,
    ):
        super().__init__()

        add_tokens = get_add_tokens(add_token)
        self.tokenizer = get_tokenizer(base, add_tokens)

        self.model_conv: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(base)  # type: ignore
        self.model_conv.resize_token_embeddings(len(self.tokenizer))  # type: ignore

        if parallel is not None:
            self.parallel = parallel
        else:
            cc = os.cpu_count()
            if cc is None:
                cc = 1
            self.parallel = cc

    def forward(
        self,
        batch,
        labels=None,
    ) -> Seq2SeqLMOutput:
        lmoutput1 = self.model_conv(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch["target_mask"],
            labels=labels,
            output_hidden_states=True,
        )
        return lmoutput1

    def _step(self, batch) -> torch.FloatTensor:
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs1 = self(
            batch=batch,
            labels=labels,
        )
        assert outputs1.loss is not None
        return_loss = outputs1.loss
        return return_loss

    def training_step(self, batch, batch_idx):
        loss: torch.FloatTensor = self._step(batch)
        self.log(
            "train_loss",
            loss,
            #             batch_size=self.args.bs,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss: torch.FloatTensor = self._step(batch)
        self.log(
            "val_loss",
            loss,
            #                 batch_size=self.args.bs_dev,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    def test_step(self):
        """Use pilota.cli"""
        raise NotImplementedError


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
        parser.link_arguments("data.base", "model.base", apply_on="instantiate")

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
        pass

    def after_fit(self):
        self.datamodule.tokenizer.save_pretrained(self.config.fit.output)

        tuner2 = T5FineTuner.load_from_checkpoint(
            self.my_checkpoint_callback.best_model_path,
            base=self.config.fit.data.base,
            ol=self.config.fit.data.ol,
            add_token=self.config.fit.data.add_token,
            parallel=self.config.fit.model.parallel,
        )
        tuner2.model_conv.save_pretrained(self.config.fit.output)  # type:ignore


def main():
    _ = MyLtCLI(T5FineTuner, MyDataModule)


if __name__ == "__main__":
    main()
