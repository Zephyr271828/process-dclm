# DCLM Processing

This repo downloads, tokenizes, shuffles, and uploads the DCLM dataset.

## Install

Use Python 3.11 or 3.12.

```bash
conda create -n dclm python=3.11 -y
conda activate dclm
python -m pip install --upgrade pip
pip install "huggingface_hub[cli]" grain tensorflow transformers zstandard tqdm array_record
```

Verify the two tools the pipeline depends on:

```bash
python -c "import grain, huggingface_hub; print('ok')"
hf --help
```

If `hf` is not on your path, reinstall `huggingface_hub` with the `cli` extra.

## What changed

- The pipeline can now run locally without `sbatch`.
- The full pipeline can be started with one command and will keep running in the background.
- `index.json` is now treated as portable. Paths are resolved from the local dataset root instead of assuming another machine's mount points.
- Each job still processes one shard at a time. No extra grouping is added.

## Quick start

Run the full pipeline in the background:

```bash
bash run_dclm_pipeline.sh --workers 4
```

Logs:

- Launcher log: `logs/run_dclm_pipeline.launch.log`
- Per-job logs: `logs/dclm_pipeline/`

## Path settings

Defaults are relative to this repo, but you can override them with environment variables:

- `DCLM_DATASET_ROOT`
- `DCLM_INDEX_PATH`
- `DCLM_TOKENIZER_NAME`
- `DCLM_TOKENIZE_OUT_DIR`
- `DCLM_MERGED_TOKEN_DIR`
- `DCLM_SHUFFLED_DIR`

## Individual stages

```bash
python dclm_pipeline.py download --workers 4
python dclm_pipeline.py tokenize --workers 4
python dclm_pipeline.py merge --workers 4
python dclm_pipeline.py upload --workers 4
```

## Rebuild the index

If you need to regenerate `index.json` for a local dataset copy:

```bash
python create_index.py
```
