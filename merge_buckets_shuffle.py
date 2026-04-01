import os
import glob
import argparse
import random
from tqdm import tqdm

from array_record.python.array_record_module import (
    ArrayRecordReader,
    ArrayRecordWriter,
)
from dclm_paths import ARRAY_RECORD_GROUP_SIZE, SHUFFLED_DIR, TOKENIZE_OUT_DIR

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(
    description="Merge same-index ArrayRecord buckets, shuffle, and write out."
)
parser.add_argument("--i", type=int, required=True, help="Bucket index to merge")
parser.add_argument(
    "--delete",
    action="store_true",
    help="Delete input bucket shards after merge (replace with empty files)",
)
parser.add_argument(
    "--batch",
    type=int,
    default=256,
    help="Read batch size (records per read call)",
)
args = parser.parse_args()

# ----------------------------
# Paths
# ----------------------------
IN_DIR = TOKENIZE_OUT_DIR
OUT_DIR = SHUFFLED_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / f"bucket_{args.i:04d}.array_record"

# Remove stale output if exists
if OUT_PATH.exists():
    OUT_PATH.unlink()

# Find all bucket shards
input_paths = sorted(
    glob.glob(str(IN_DIR / "*" / f"bucket_{args.i:04d}.array_record"))
)

print(f"🔀 Merging bucket {args.i}")
print(f"📥 Found {len(input_paths)} shards")
print(f"📤 Output → {OUT_PATH}")

if not input_paths:
    raise RuntimeError(f"No input buckets found for bucket {args.i}")

# ----------------------------
# Load all records (CORRECT API)
# ----------------------------
records = []
total_records = 0

for path in input_paths:
    reader = ArrayRecordReader(path)
    n = reader.num_records()
    total_records += n

    for start in tqdm(range(0, n, args.batch), desc=f"Reading {os.path.basename(path)}"):
        end = min(start + args.batch, n)
        batch = reader.read(start, end)   # ✅ list[bytes]

        for rec in batch:
            # Some builds return memoryview
            if isinstance(rec, memoryview):
                rec = rec.tobytes()
            records.append(rec)

print(f"📦 Loaded {len(records)} records (expected {total_records})")

assert len(records) == total_records, "Record count mismatch!"

# ----------------------------
# Shuffle
# ----------------------------
random.seed(42)
random.shuffle(records)

# ----------------------------
# Write merged bucket
# ----------------------------
writer = ArrayRecordWriter(str(OUT_PATH), ARRAY_RECORD_GROUP_SIZE)
for rec in records:
    writer.write(rec)
writer.close()

print(f"✅ Wrote {len(records)} records → {OUT_PATH}")

# ----------------------------
# Optional cleanup
# ----------------------------
if args.delete:
    for path in input_paths:
        try:
            os.remove(path)
            open(path, "wb").close()  # replace with empty placeholder
        except OSError as e:
            raise RuntimeError(f"Failed to clean up {path}: {e}")

    print(f"🧹 Cleaned up {len(input_paths)} input shards")

print("🎉 Done.")
