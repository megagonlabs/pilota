
# plain2request

To make [input JSON-lines for Pilota](format.md) easily, you can use ``python -m pilota.convert.plain2request``.
This converts TSV (Tab-Separated Values) format file to JSON-lines.

- Dialog

    ```tsv
    ご要望をお知らせください[TAB]はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。
    ```

- Review (no tabs; single column)

    ```tsv
    お正月に泊まりました。夜景が綺麗でした。
    ```
