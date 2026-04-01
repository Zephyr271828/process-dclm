import os
import subprocess
import argparse
from dclm_paths import SHUFFLED_DIR

parser = argparse.ArgumentParser(
    description="Merge same-index buckets across array_record shards, shuffle, and write out."
)
parser.add_argument("--i", type=int, required=True, help="Bucket index to merge and shuffle")
args = parser.parse_args()

OUT_DIR = SHUFFLED_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / f"bucket_{args.i:04d}.array_record"

GCS_PREFIXES = [
    "gs://llm_pruning_us_central2_b/datasets/dclm/llama2_array_record_with_special_tokens_1T",
    "gs://llm_pruning_us_central1/datasets/dclm/llama2_array_record_with_special_tokens_1T",
    "gs://llm_pruning_us_east5/datasets/dclm/llama2_array_record_with_special_tokens_1T",
]


for prefix in GCS_PREFIXES:
    dst = f"{prefix}/{OUT_PATH.name}"
    print(f"⬆️ Uploading {OUT_PATH} → {dst}")
    subprocess.run(
        ["gsutil", "-m", "cp", str(OUT_PATH), dst],
        check=True,
    )

print("✅ Upload complete to all buckets.")
