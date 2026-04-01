import glob
import argparse
import tensorflow as tf
from tqdm import tqdm
from array_record.python.array_record_module import ArrayRecordReader

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(
    description="Fast token counting for ArrayRecord files (single-record probe)"
)
# parser.add_argument(
#     "--glob",
#     type=str,
#     required=True,
#     help="Glob pattern for ArrayRecord files",
# )
args = parser.parse_args()
args.glob = "/n/fs/vision-mix/yx1168/pruning/datasets/dclm/llama2-array-record-w-special-tokens-shuffled/*"

# ----------------------------
# Helpers
# ----------------------------
def parse_ids(serialized: bytes):
    """Extract input_ids from a serialized TF Example"""
    if isinstance(serialized, memoryview):
        serialized = serialized.tobytes()

    ex = tf.train.Example()
    ex.ParseFromString(serialized)
    return ex.features.feature["input_ids"].int64_list.value

# ----------------------------
# Main
# ----------------------------
paths = sorted(glob.glob(args.glob))
assert paths, f"No files matched glob: {args.glob}"

grand_total_tokens = 0
grand_total_chunks = 0

for i, path in enumerate(tqdm(paths, desc="Files")):
    reader = ArrayRecordReader(path)

    # number of chunks in file
    num_records = reader.num_records()
    assert num_records > 0, f"Empty file: {path}"

    # read exactly ONE record
    record0 = reader.read(0, 1)[0]
    tokens_per_chunk = len(parse_ids(record0))

    file_tokens = num_records * tokens_per_chunk

    grand_total_chunks += num_records
    grand_total_tokens += file_tokens

    print(
        f"[FILE] {path} | "
        f"chunks={num_records:,} | "
        f"tokens/chunk={tokens_per_chunk:,} | "
        f"tokens={file_tokens:,}"
    )
    
    if i > 10:
        break

print("\n================ SUMMARY ================")
print(f"Total files   : {len(paths):,}")
print(f"Total chunks  : {grand_total_chunks:,}")
print(f"Total tokens  : {grand_total_tokens:.2e}")
print("========================================")