#!/usr/bin/env python3

import argparse
import string
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import tqdm
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from pilota.predict import Predictor


def tokenize(path_in: Path, tokenizer: PreTrainedTokenizer, path_out: Path) -> None:
    tid2t = {v: k for k, v in tokenizer.get_vocab().items()}
    with path_in.open() as inf, path_out.open("w") as outf:
        for line in inf:
            sent: str = Predictor.normalize(line.strip())
            tids = tokenizer.encode(sent)  # type: ignore

            char_idx: int = 0
            token_count: int = 0
            ts = []
            ids = []
            for tid in tids:
                t = tid2t[tid]
                if t.startswith("<0x"):
                    ts.append(t)
                    ids.append(tid)
                    continue
                if len(ts) > 0:
                    outf.write(f"#{token_count}-#{token_count+len(ts)}\t{sent[char_idx]}\t{ids}\t{''.join(ts)}\n")
                    char_idx += 1
                    token_count += len(ts)
                    ts = []
                    ids = []
                outf.write(f"#{token_count}\t{t}\t{tid}\n")
                token_count += 1
                char_idx += len(t)
            if len(ts) > 0:
                outf.write(f"#{token_count}-#{token_count+len(ts)}\t{sent[char_idx]}\t{ids}\t{''.join(ts)}\n")


def get_unk_words(path_in: Path, tokenizer: PreTrainedTokenizer, path_out: Path) -> None:
    counter: DefaultDict[str, int] = defaultdict(int)
    with path_in.open() as inf:
        for line in tqdm.tqdm(inf):
            items = Predictor.normalize(line.strip()).split()
            cnt: int = int(items[0])
            sent = items[1]
            d = tokenizer(sent, return_offsets_mapping=True)
            for ti, tspan in zip(d["input_ids"], d["offset_mapping"]):  # type: ignore
                if ti == tokenizer.unk_token_id:
                    phrase: str = sent[tspan[0] : tspan[1]]
                    for c in phrase:
                        counter[c] += cnt
    for c, n in counter.items():
        print(f"{c}\t{n}")

    vocab = tokenizer.get_vocab()
    for g in [
        string.ascii_letters,
        string.digits,
        #               [chr(i) for i in range(12353, 12436)],  # hiragana
        #               [chr(i) for i in range(12449, 12532 + 1)],  # katakana
    ]:
        for char in g:
            if char not in vocab and char not in counter:
                print(f"{char}\t0")
                tokenizer.add_tokens(char)


def list_vocab(tokenizer: PreTrainedTokenizer, path_out: Path):
    with path_out.open("w") as outf:
        for k, v in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
            outf.write(f"{k}\t{v}\n")


def get_opts() -> argparse.Namespace:
    oparser = argparse.ArgumentParser()
    oparser.add_argument("--input", "-i", type=Path, default="/dev/stdin", required=False)
    oparser.add_argument("--tokenizer", "-t", required=True)
    oparser.add_argument("--list", action="store_true")
    oparser.add_argument("--unk", action="store_true")
    oparser.add_argument("--output", "-o", type=Path, default="/dev/stdout", required=False)
    return oparser.parse_args()


def main() -> None:
    opts = get_opts()
    tokenizer = T5Tokenizer.from_pretrained(opts.tokenizer)
    if opts.list:
        list_vocab(tokenizer, opts.output)
    elif opts.unk:
        get_unk_words(opts.input, tokenizer, opts.output)
    else:
        tokenize(opts.input, tokenizer, opts.output)


if __name__ == "__main__":
    main()
