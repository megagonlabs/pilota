#!/usr/bin/env python3


import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

try:
    from sumeval.metrics.rouge import RougeCalculator
except ImportError:
    raise Exception("You need to install the extra 'train'")


from pilota.const.metachar import NONE
from pilota.evaluate.schema import RougeScore
from pilota.predict import Predictor


def trim(orig: str) -> str:
    return orig.replace("【", "").replace("】", "")


def get_tokenized(
    sentences: list[str],
    rouge: RougeCalculator,
) -> list[list[str]]:
    # Use _lang to avoid tokenize_with_preprocess()
    items_sent_tokenized: list[list[str]] = [rouge._lang.tokenize(trim(sent)) for sent in sentences]
    if len(items_sent_tokenized) == 0:
        items_sent_tokenized = [[NONE]]
    return items_sent_tokenized


def get_rouge_score(
    items_gold_tokenized: list[list[str]],
    items_system_tokenized: list[list[str]],
    rouge: RougeCalculator,
) -> RougeScore:
    tokenized_gold: list[str] = list(itertools.chain.from_iterable(items_gold_tokenized))
    tokenized_system: list[str] = list(itertools.chain.from_iterable(items_system_tokenized))

    rouge_1 = rouge.rouge_n(summary=tokenized_system, references=[tokenized_gold], n=1)
    rouge_2 = rouge.rouge_n(summary=tokenized_system, references=[tokenized_gold], n=2)
    rouge_l = rouge.rouge_l(summary=tokenized_system, references=[tokenized_gold])
    return RougeScore(
        rouge_1=rouge_1,
        rouge_2=rouge_2,
        rouge_l=rouge_l,
    )


rouge = RougeCalculator(lang="ja", stopwords=False, stemming=False)


def _myrouge(
    target_predicted: str,
    target_gold: str,
) -> float:
    items_gold: list[str] = Predictor.decode_to_scuds(target_gold)
    items_system: list[str] = Predictor.decode_to_scuds(target_predicted)

    items_gold_tokenized: list[list[str]] = get_tokenized(items_gold, rouge)
    items_system_tokenized: list[list[str]] = get_tokenized(items_system, rouge)

    tokenized_gold: list[str] = list(itertools.chain.from_iterable(items_gold_tokenized))
    tokenized_system: list[str] = list(itertools.chain.from_iterable(items_system_tokenized))
    rouge_l: float = rouge.rouge_l(summary=tokenized_system, references=[tokenized_gold])
    return rouge_l


def get_avg_rouge(
    *,
    iter_query: Iterator[tuple[str, str]],
    num_parallel: int,
):
    pool = ProcessPoolExecutor(max_workers=num_parallel)
    with pool as executor:
        tasks = [
            executor.submit(
                _myrouge,
                target_predicted,
                target_gold,
            )
            for (target_predicted, target_gold) in iter_query
        ]

    rouge_ls = [future.result() for future in as_completed(tasks)]
    return sum(rouge_ls) / len(rouge_ls)
