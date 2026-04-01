import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


DATASET_ROOT = env_path("DCLM_DATASET_ROOT", REPO_ROOT / "dclm_subset")
INDEX_PATH = env_path("DCLM_INDEX_PATH", REPO_ROOT / "index.json")
TOKENIZER_NAME = os.environ.get(
    "DCLM_TOKENIZER_NAME",
    "/n/fs/vision-mix/yx1168/model_ckpts/Llama-2-7b-hf/",
)

TOKENIZE_OUT_DIR = env_path("DCLM_TOKENIZE_OUT_DIR", REPO_ROOT / "llama2-bucket-pieces")
MERGED_TOKEN_DIR = env_path(
    "DCLM_MERGED_TOKEN_DIR",
    REPO_ROOT / "llama2-array-record-w-special-tokens",
)
SHUFFLED_DIR = env_path(
    "DCLM_SHUFFLED_DIR",
    REPO_ROOT / "llama2-array-record-w-special-tokens-shuffled",
)

ARRAY_RECORD_GROUP_SIZE = os.environ.get("DCLM_ARRAY_RECORD_GROUP_SIZE", "group_size:1")
