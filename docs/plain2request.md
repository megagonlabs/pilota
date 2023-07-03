
# plain2request

To make [input JSON-lines for Pilota](format.md) easily, you can use ``python -m pilota.convert.plain2request``.
This converts TSV (Tab-Separated Values) format file to JSON-lines.

The last column is the target of the SCUD generation.
(Replace ``[TAB]`` with the tab character.)

1. No context
    - Input

        ```txt
        部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいです。
        ```

    - Output

        ```json
        {
          "context": null,
          "utterance": "部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいです。",
          "sentences": null,
          "meta": {}
        }
        ```

2. With an agent utterance as context
    - Input

        ```txt
        ご要望をお知らせください[TAB]はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。
        ```

    - Output

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
          "meta": {}
        }
        ```

3. With some utterances as context
    - Input

        ```txt
        来月の連休に旅行したいと考えていて[TAB]どのエリアでお考えでしょうか?[TAB]四国で考えています。
        ```

    - Output

        ```json
        {
          "context": [
            {
              "name": "user",
              "text": "来月の連休に旅行したいと考えていて"
            },
            {
              "name": "agent",
              "text": "どのエリアでお考えでしょうか?"
            }
          ],
          "utterance": "四国で考えています。",
          "sentences": null,
          "meta": {}
        }
        ```
