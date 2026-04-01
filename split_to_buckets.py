import os
import pdb
import argparse
import hashlib
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter
from dclm_paths import ARRAY_RECORD_GROUP_SIZE, MERGED_TOKEN_DIR, TOKENIZE_OUT_DIR

NUM_BUCKETS = 128
OUT_DIR = TOKENIZE_OUT_DIR

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--i", type=int, required=True, help="Merged array_record file")
args = parser.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def parse_example(serialized):
    ex = tf.train.Example()
    ex.ParseFromString(serialized)
    return ex

def hash_example(ex):
    ids = ex.features.feature["input_ids"].int64_list.value
    h = hashlib.blake2b(
        bytes(",".join(map(str, ids)), encoding="utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")

# ----------------------------
# Writers
# ----------------------------
writers = {}
bucket_dir = OUT_DIR / f"array_record_{args.i:04d}"
bucket_dir.mkdir(parents=True, exist_ok=True)
for b in range(NUM_BUCKETS):
    writers[b] = ArrayRecordWriter(
        str(bucket_dir / f"bucket_{b:04d}.array_record"),
        ARRAY_RECORD_GROUP_SIZE,
    )

# ----------------------------
# Main loop
# ----------------------------
input_path = MERGED_TOKEN_DIR / f"dclm.merged.{args.i:04d}.array_record"
reader = ArrayRecordReader(str(input_path))

count = 0
# for record in reader:
for record in reader.read():
    ex = parse_example(record)
    b = hash_example(ex) % NUM_BUCKETS
    writers[b].write(record)
    count += 1

for w in writers.values():
    w.close()

print(f"✅ Split {count} entries into {NUM_BUCKETS} buckets")

# try:
#     os.remove(input_path)
#     open(input_path, "wb").close()
#     print(f"🗑️ Deleted source file: {input_path}")
# except OSError as e:
#     raise RuntimeError(f"Failed to delete {input_path}: {e}")
