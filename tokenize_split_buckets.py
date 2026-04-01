import os
import io
import json
import argparse
import shutil
import subprocess
import hashlib
import zstandard as zstd
from tqdm import tqdm

import tensorflow as tf
from transformers import AutoTokenizer
from array_record.python.array_record_module import ArrayRecordWriter

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(
    description="Tokenize JSONL.ZST files and directly write to bucketed ArrayRecord files."
)
parser.add_argument("--i", type=int, required=True, help="Index of the K-file group to process")
args = parser.parse_args()

# ----------------------------
# Config
# ----------------------------
K = 32
SEQ_LEN = 4096
NUM_BUCKETS = 128

TOKENIZER_NAME = "/n/fs/vision-mix/yx1168/model_ckpts/Llama-2-7b-hf/"
ROOT_DIR = "/n/fs/vision-mix/yx1168/pruning/datasets/dclm/dclm_subset"
INDEX_PATH = "/n/fs/vision-mix/yx1168/pruning/datasets/dclm/scripts/index.json"

OUT_DIR = "/n/fs/vision-mix/yx1168/pruning/datasets/dclm/llama2-bucket-pieces"
BUCKET_DIR = os.path.join(OUT_DIR, f"array_record_{args.i:04d}")

shutil.rmtree(BUCKET_DIR, ignore_errors=True)

JSON_KEY = "text"

os.makedirs(BUCKET_DIR, exist_ok=True)

# ----------------------------
# Load index
# ----------------------------
with open(INDEX_PATH, "r") as f:
    indices = json.load(f)

paths = indices[str(args.i)]
assert len(paths) == K, f"Expected {K} files, got {len(paths)}"

print("Input files:")
for p in paths:
    print(" ", p)

# ----------------------------
# Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

print(f"BOS={bos_id}, EOS={eos_id}")

# ----------------------------
# Helpers
# ----------------------------
def iter_jsonl_zst(path):
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    yield json.loads(line)

def tokenize_with_single_bos_eos(text):
    ids = tokenizer.encode(text, add_special_tokens=False)

    if bos_id is not None and ids and ids[0] == bos_id:
        ids = ids[1:]
    if eos_id is not None and ids and ids[-1] == eos_id:
        ids = ids[:-1]

    if bos_id is not None:
        ids = [bos_id] + ids
    if eos_id is not None:
        ids = ids + [eos_id]

    return ids

def hash_tokens(token_ids):
    h = hashlib.blake2b(
        bytes(",".join(map(str, token_ids)), encoding="utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")

def write_example(writer, token_ids):
    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "input_ids": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=token_ids)
                )
            }
        )
    )
    writer.write(ex.SerializeToString())

# ----------------------------
# Open bucket writers
# ----------------------------
writers = {
    b: ArrayRecordWriter(os.path.join(BUCKET_DIR, f"bucket_{b:04d}.array_record"))
    for b in range(NUM_BUCKETS)
}

# ----------------------------
# Main loop
# ----------------------------
buffer = []
total_written = 0

for path in paths:
    print(f"\nProcessing {path}")

    if not os.path.exists(path) or os.path.getsize(path) < 64 * 1024:
        local_path = path[path.find("global"):]
        cmd = (
            "huggingface-cli download mlfoundations/dclm-baseline-1.0 "
            "--repo-type dataset "
            f"--include '{local_path}' "
            "--local-dir dclm_subset "
            "--local-dir-use-symlinks False"
        )
        subprocess.run(cmd, shell=True, check=True)

    for obj in tqdm(iter_jsonl_zst(path)):
        text = obj.get(JSON_KEY, "")
        if not text:
            continue

        ids = tokenize_with_single_bos_eos(text)
        buffer.extend(ids)

        while len(buffer) >= SEQ_LEN:
            chunk = buffer[:SEQ_LEN]
            buffer = buffer[SEQ_LEN:]

            bucket = hash_tokens(chunk) % NUM_BUCKETS
            write_example(writers[bucket], chunk)
            total_written += 1

# Flush remainder
if buffer:
    bucket = hash_tokens(buffer) % NUM_BUCKETS
    write_example(writers[bucket], buffer)
    total_written += 1

# ----------------------------
# Close writers
# ----------------------------
for w in writers.values():
    w.close()

print(f"\n✅ Done. Wrote {total_written} records into {NUM_BUCKETS} buckets")
print("Output dir:", BUCKET_DIR)