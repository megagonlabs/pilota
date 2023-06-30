
# I/O Format

The format of classes is defined in [``pilota.schema``](https://github.com/megagonlabs/pilota/blob/master/pilota/schema.py).
Inputs and outputs of Pilota are JSON-lines of those classes.

**Note: All examples are shown on multiple lines for clarity, but should be written on a single line when used in practice.**

## Input format: ``pilota.schema.Request``

- ``context`` can be ``null``
- ``utterance`` and ``sentences`` must be ``null`` on one side and have a value on the other

### Conversion from TSV

You can get input JSON-lines from a TSV (Tab-Separated Values) format file by using ``python -m pilota.convert.plain2request``

- Dialog

    ```tsv
    ご要望をお知らせください[TAB]はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。
    ```

- Review (no tabs; single column)

    ```tsv
    お正月に泊まりました。夜景が綺麗でした。
    ```

### Examples of input

- Context and utterance

  ```json
    {
        "context": [
            {
                "name": "agent",
                "text": "ご要望をお知らせください"
            }
        ],
        "utterance": "はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。",
        "sentences": null,
    }
  ```

- Context and a list of sentences

  ```json
    {
        "context": [
            {
                "name": "agent",
                "text": "ご利用ありがとうございます。"
            },
            {
                "name": "user",
                "text": "よろしくおねがいします"
            },
            {
                "name": "agent",
                "text": "ご要望をお知らせください"
            }
        ],
        "utterance": null,
        "sentences": [
            "はい。",
            "部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。"
        ],
    }
  ```

- Review text. ``context`` should be ``null``.

  ```json
    {
        "context": null,
        "utterance": "お正月に泊まりました。夜景が綺麗でした。",
        "sentences": null,
    }
  ```

## Output format: ``list`` of ``pilota.schema.ResultForSentence``

- ``sentence``: The target sentence for SCUD generation
- ``scuds_nbest``:  N-best of SCUDS
- ``scores``: Score of each candidate of N-best
- ``scores_detail``: Detail scores of each candidate of N-best

### Example of output

```json
[
  {
    "scuds_nbest": [
      [],
      [
        "【customer】が希望する。"
      ]
    ],
    "original_ranks": [
      0,
      1
    ],
    "scores": [
      0.9912104725837707,
      0.15555430445820093
    ],
    "scores_detail": [
      {
        "OK": 0.9707015752792358,
        "incorrect_none": 0.043491218239068985,
        "lack": 0.0007595779024995863,
        "limited": 0.0003174646117258817,
        "non_fluent": 0.00024337830836884677,
        "untruth": 0.0012659059138968587
      },
      {
        "OK": 0.018514348194003105,
        "incorrect_none": 0.0038577320519834757,
        "lack": 0.978103518486023,
        "limited": 0.001594067900441587,
        "non_fluent": 0.0016208436572924256,
        "untruth": 0.010369746945798397
      }
    ],
    "sentence": "はい。"
  },
  {
    "scuds_nbest": [
      [
        "部屋から富士山が見えるホテルが良い。",
        "夜景を見ながら食事のできるホテルが良い。"
      ],
      [
        "部屋から富士山が見えるホテルが良い。",
        "夜景を見ながら食事のできるホテルがよい。"
      ]
    ],
    "original_ranks": [
      0,
      1
    ],
    "scores": [
      0.9952571213245391,
      0.8453719913959503
    ],
    "scores_detail": [
      {
        "OK": 0.9841904044151306,
        "incorrect_none": 0.011234153062105179,
        "lack": 0.0030710017308592796,
        "limited": 0.00038350545219145715,
        "non_fluent": 0.0002909886243287474,
        "untruth": 0.003009898355230689
      },
      {
        "OK": 0.9845733046531677,
        "incorrect_none": 0.011145127937197685,
        "lack": 0.0029782589990645647,
        "limited": 0.00038316904101520777,
        "non_fluent": 0.00029046559939160943,
        "untruth": 0.0029817328322678804
      }
    ],
    "sentence": "部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。"
  }
]
```
