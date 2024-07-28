## Install

```shell
uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt

# in development mode
uv pip install -r requirements-dev.txt
```

## Download audio

```shell
wget https://github.com/egorsmkv/wav2vec2-uk-demo/raw/master/short_1_16k.wav
```

## Run

```shell
python test_hidet.py
```
