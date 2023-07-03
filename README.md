
# Pilota: SCUD generator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/megagonlabs/pilota/actions/workflows/ci.yml/badge.svg)](https://github.com/megagonlabs/pilota/actions/workflows/ci.yml)
[![Typos](https://github.com/megagonlabs/pilota/actions/workflows/typos.yml/badge.svg)](https://github.com/megagonlabs/pilota/actions/workflows/typos.yml)

| Name | Input Utterance | Output SCUD |
| --- | --- | --- |
| Agent | ご要望をお知らせください | - |
| User | はい。 | (none) |
| | 部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。 | 部屋から富士山が見るホテルが良い。<br>夜景を見ながら食事のできるホテルが良い。|

## Quick start

### Install

```bash
pip install -U 'pilota[ja_line] @ git+https://github.com/megagonlabs/pilota'
```

If you need compatible torch for your GPU, please install the specific package like the following step.
Please read <https://pytorch.org/>.

```bash
pip install -U torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Run

1. Prepare inputs ([Input Format](docs/format.md) and [plain2request](docs/plain2request.md))
    - Command

        ```bash
        echo -e 'ご要望をお知らせください\tはい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。\nこんにちは\tこんにちは' | python -m pilota.convert.plain2request | tee input.jsonl
        ```

    - Output

        ```jsonl
        {"context": [{"name": "agent", "text": "ご要望をお知らせください"}], "utterance": "はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。", "sentences": null, "meta": {}}
        {"context": [{"name": "agent", "text": "こんにちは"}], "utterance": "こんにちは", "sentences": null, "meta": {}}
        ```

2. Feed it to Pilota
    - Command

        ```console
        pilota -m megagonlabs/pilota_dialog --batch_size 1 --outlen 60 --nbest 1 --beam 5 < input.jsonl
        ```

    - Output

        ```jsonl
        [{"scuds_nbest": [[]], "original_ranks": [0], "scores": [0.9911208689212798], "scores_detail": [{"OK": 0.9704028964042664, "incorrect_none": 0.04205145686864853, "lack": 0.0007874675211496651, "limited": 0.0003119863977190107, "non_fluent": 0.0002362923405598849, "untruth": 0.0013080810895189643}], "sentence": "はい。"}, {"scuds_nbest": [["部屋から富士山が見えるホテルが良い。", "夜景を見ながら食事のできるホテルが良い。"]], "original_ranks": [0], "scores": [0.9952289938926696], "scores_detail": [{"OK": 0.9840966463088989, "incorrect_none": 0.010280555114150047, "lack": 0.0032871251460164785, "limited": 0.00041511686868034303, "non_fluent": 0.0002954243100248277, "untruth": 0.003289491171017289}], "sentence": "部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。"}]
        [{"scuds_nbest": [[]], "original_ranks": [0], "scores": [0.9831213414669036], "scores_detail": [{"OK": 0.9704028964042664, "incorrect_none": 0.04205145686864853, "lack": 0.0007874675211496651, "limited": 0.0003119863977190107, "non_fluent": 0.0002362923405598849, "untruth": 0.0013080810895189643}], "sentence": "こんにちは"}]
        ```

`-m` option also accepts paths of local models.

```bash
pilota -m /path/to/model --batch_size 1 --ol 60 < input.jsonl
```

Check other options by ``pilota -h``.

## Models

Models are available on <https://huggingface.co/megagonlabs/>.

| Model | Input Context | Input Utterance | Output |
| --- | --- | --- | --- |
| [megagonlabs/pilota_dialog](https://huggingface.co/megagonlabs/pilota_dialog) | Dialogue between a user looking for accommodation and an agent | User's last utterance | SCUDs |
| [megagonlabs/pilota_scud2query](https://huggingface.co/megagonlabs/pilota_scud2query) | (Not required) | Users's SCUDs | Queries for accommodation search |
| [megagonlabs/pilota_hotel_review](https://huggingface.co/megagonlabs/pilota_hotel_review) | (Not required) | Text of an accommodation review | SCUDs |

Once downloaded, the model will not be downloaded again.
If you cancel the download of a model halfway through the first start-up, or if you need to update it to the latest version, please run with ``--check_model_update``.

You can check local path of downloaded models.

```bash
huggingface-cli scan-cache | grep ^megagonlabs
```

## [Documents](docs)

- [Format](docs/format.md)
- [plain2request](docs/plain2request.md)
- [Web API server](docs/web_api.md)
- [Training](docs/training.md)

## References

1. Yuta Hayashibe.
    Self-Contained Utterance Description Corpus for Japanese Dialog.
    Proc of LREC, pp.1249-1255. (LREC 2022)
    [[PDF]](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.133.pdf)
2. 林部祐太．
    要約付き宿検索対話コーパス．
    言語処理学会第27回年次大会論文集，pp.340-344. 2021. (NLP 2021)
    [[PDF]](https://www.anlp.jp/proceedings/annual_meeting/2021/pdf_dir/P2-5.pdf)

## License

Apache License 2.0
