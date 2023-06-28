
# API server

You need an additional package `web`.

```bash
pip install -U 'git+https://github.com/megagonlabs/pilota@main#egg=pilota[web]'
```

## Start a server

```console
# Optional: If you use dynamic model update, set ADMIN_KEY to the environment variable
export ADMIN_KEY=$(uuidgen |tee /dev/stderr)

# Run
python3 -m pilota.web \
    --name my_model_name \
    --model /path/to/model \
    --name my_model_name2 \
    --model /path/to/model2 \
    --port 7001 \
    --root_path /app/pilota
```

- You can use several models
    - The first model is default
- ``root_path`` is a parameter for uvicorn (Optional)

## How to use

- Browser interface:  <http://0.0.0.0:7001/>
- All inputs will be NFKC normalized
- For sentences in an utterance

```console
$ curl -X 'POST' 'http://0.0.0.0:7001/api/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "context": [
    { "speaker": "agent", "text": "ご要望はありますか？" }
  ],
  "sentences": ["近くにコンビニはありますか？", "あと部屋に冷蔵庫があるといいです。"],
  "model_name": "default"
}'
{"scuds":[["近くにコンビニがあるか【customer】が知りたい。"],["部屋に冷蔵庫があるとよい。"]],"sentences":["近くにコンビニはありますか？","あと部屋に冷蔵庫があるといいです。"]}
```

- For an utterance

```console
$ curl -X 'POST' 'http://0.0.0.0:7001/api/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "context": [
    { "speaker": "agent", "text": "ご要望はありますか？" }
  ],
  "utterance": "近くにコンビニはありますか？あと部屋に冷蔵庫があるといいです。",
}'
{"scuds":[["近くにコンビニがあるか【customer】が知りたい。"],["部屋に冷蔵庫があるとよい。"]],"sentences":["近くにコンビニはありますか？","あと部屋に冷蔵庫があるといいです。"]}

$ curl -X 'POST' 'http://0.0.0.0:7001/api/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "context": [
    { "speaker": "agent", "text": "こんにちは" }
  ],
  "utterance": "こんにちは!"
}'
{"scuds":[[]],"sentences":["こんにちは!"]}

$ curl -X 'POST' 'http://0.0.0.0:7001/api/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "context": [
    { "speaker": "agent", "text": "ご要望はありますか？" }
  ],
  "utterance": "近くにコンビニはありますか？あと部屋に冷蔵庫があって夜景が見えるといいです。"
}'
{"scuds":[["コンビニの近くにコンビニがあるか【customer】が知りたい。"],["部屋に冷蔵庫があると良い。","夜景が見えると良い。"]],"sentences":["近くにコンビニはありますか?","あと部屋に冷蔵庫があって夜景が見えるといいです。"]}
```

## Dynamic model update

```console
# Load a new model
curl -X POST -H "Content-Type: application/json" -d ' {"model_name":"new_model", "path":"/path/to/new_model/", "new_default": "new_model", "admin_key": "XXXXXXX"} ' 0.0.0.0:7001/api/admin/models

# Remove a model
curl -X POST -H "Content-Type: application/json" -d ' {"model_name":"some_model", "admin_key": "XXXXXXX"} ' 0.0.0.0:7001/api/admin/models

# Change default
curl -X POST -H "Content-Type: application/json" -d ' {"new_default":"a_model", "admin_key": "XXXXXXX"} ' 0.0.0.0:7001/api/admin/models
```
