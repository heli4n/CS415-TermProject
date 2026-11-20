# CS415-TermProject

Artist dependence in music genre classification (FMA-small).

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Log-mel cache (ECS-17)

```bash
python scripts/cache_logmel.py \
    --audio-dir /content/drive/MyDrive/cs415/raw/fma_small \
    --out-dir   /content/drive/MyDrive/cs415/cache/logmel \
    --bad-files /content/drive/MyDrive/cs415/cache/bad_files.csv
```

Already-cached tracks are skipped, so the same command can simply be re-run after a Colab
disconnect. Use `--limit 20` for a quick smoke test. Formats: `docs/data_contract.md`.
