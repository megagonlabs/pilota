
# Pilota: SCUD generator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![CI](https://github.com/megagonlabs/pilota/actions/workflows/ci.yml/badge.svg)](https://github.com/megagonlabs/pilota/actions/workflows/ci.yml)
[![Typos](https://github.com/megagonlabs/pilota/actions/workflows/typos.yml/badge.svg)](https://github.com/megagonlabs/pilota/actions/workflows/typos.yml)
[![markdownlint](https://img.shields.io/badge/markdown-lint-lightgrey)](https://github.com/markdownlint/markdownlint)
[![jsonlint](https://img.shields.io/badge/json-lint-lightgrey)](https://github.com/dmeranda/demjson)
[![yamllint](https://img.shields.io/badge/yaml-lint-lightgrey)](https://github.com/adrienverge/yamllint)

| Name | Utterance | SCUD |
| --- | --- | --- |
| Agent | ご要望をお知らせください | - |
| User | はい。 | (none) |
| | 部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。 | 部屋から富士山が見るホテルが良い。<br>夜景を見ながら食事のできるホテルが良い。|

## Quick start

### Install

```bash
pip install -U git+https://github.com/megagonlabs/pilota@main
```

If you need compatible torch for your GPU, please install the specific package like the following step.
Please read <https://pytorch.org/>.

```bash
pip install -U torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Run

Prepare inputs.

```console
$ echo -e 'ご要望をお知らせください\tはい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。\nこんにちは\tはい、こんにちは' \
    | python -m pilota.convert.plain2request | tee input.jsonl
{"context": [{"name": "agent", "text": "ご要望をお知らせください"}], "utterance": "はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。", "sentences": null, "meta": {}}
{"context": [{"name": "agent", "text": "こんにちは"}], "utterance": "はい、こんにちは", "sentences": null, "meta": {}}
```

Designate a model on <https://huggingface.co/megagonlabs/> like this.

- [megagonlabs/pilota_dialog](https://huggingface.co/megagonlabs/pilota_dialog)
- [megagonlabs/pilota_scud2query](https://huggingface.co/megagonlabs/pilota_scud2query)
- [megagonlabs/pilota_hotel_review](https://huggingface.co/megagonlabs/pilota_hotel_review)

```bash
pilota -m megagonlabs/pilota_dialog --batch_size 1 --ol 60 < input.jsonl
```

You can designate other local models.

```bash
pilota -m /path/to/model --batch_size 1 --ol 60 < input.jsonl
```

Check other options by ``pilota -h``.

### Input format

- JSON-lines of [``pilota.schema.Request``](https://github.com/megagonlabs/pilota/blob/master/pilota/schema.py)

## [Documents](docs)

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
