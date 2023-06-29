# Training

You need an additional package `train`.

```bash
pip install -U 'git+https://github.com/megagonlabs/pilota@main#[train]'
```

## Training for dialogs

- Needed corpus
    - [asdc](https://github.com/megagonlabs/asdc)
    - (optional, internal only) ``scud_internal``

```bash
OUTPUT=/path/to/output
make -j1 -f ./train.mk \
    OUTPUT="${OUTPUT}" \
    T5BASE=megagonlabs/t5-base-japanese-web-8k \
    BATCH=100 BATCH_DEV=100 EPOCH=20 IN_LEN=128 OUT_LEN=64 BATCH_PRED=100 \
    all
```

## Training for Jalan reviews

- Needed corpus
    - [Hotel Review SCUD](https://github.com/megagonlabs/hotel_review_scud)

```bash
OUTPUT=/path/to/output
make -j1 -f ./train.mk \
    OUTPUT="${OUTPUT}" \
    T5BASE=megagonlabs/t5-base-japanese-web-8k \
    BATCH=86 BATCH_DEV=112 EPOCH=20 IN_LEN=128 OUT_LEN=64 BATCH_PRED=120 JALAN=1 \
    all
```

## Training for Scud2Query

- Needed Corpus
    - [Scud to Query](https://github.com/megagonlabs/scud2query)

```bash
OUTPUT=/path/to/output
make -j1 -f ./train.mk \
    OUTPUT="${OUTPUT}" \
    T5BASE=megagonlabs/t5-base-japanese-web-8k \
    BATCH=86 BATCH_DEV=112 EPOCH=20 IN_LEN=128 OUT_LEN=64 BATCH_PRED=120 SCUD2QUERY=1 \
    all
```
