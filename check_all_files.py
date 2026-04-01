import glob
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer
from array_record.python.array_record_module import ArrayRecordReader
from dclm_paths import TOKENIZER_NAME, TOKENIZE_OUT_DIR
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
vocab_size = tokenizer.vocab_size
print(f"BOS id={bos_id}, EOS id={eos_id}, vocab_size={vocab_size}")

def parse_ids(serialized: bytes):
    ex = tf.train.Example()
    ex.ParseFromString(serialized)
    return ex.features.feature["input_ids"].int64_list.value

all_paths = glob.glob(str(TOKENIZE_OUT_DIR / "*" / "*"))

BATCH = 256  # adjust for speed/memory

for path in tqdm(sorted(all_paths)):
    try:
        reader = ArrayRecordReader(path)
        n = reader.num_records()

        min_id = None
        max_id = None
        bos_cnt = 0
        eos_cnt = 0

        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            recs = reader.read(start, end)  # list[bytes]

            for rec in tqdm(recs):
                # some bindings might return memoryview
                if isinstance(rec, memoryview):
                    rec = rec.tobytes()

                ids = parse_ids(rec)
                if not ids:
                    continue

                lo = min(ids)
                hi = max(ids)
                min_id = lo if min_id is None else min(min_id, lo)
                max_id = hi if max_id is None else max(max_id, hi)

                if bos_id is not None:
                    bos_cnt += sum(t == bos_id for t in ids)
                if eos_id is not None:
                    eos_cnt += sum(t == eos_id for t in ids)

        flags = []
        if min_id is not None and min_id < 0:
            flags.append("NEGATIVE_TOKEN")
        if max_id is not None and max_id >= vocab_size:
            flags.append("TOKEN_OOB")
        flag_str = (" | FLAGS=" + ",".join(flags)) if flags else ""

        print(f"[OK] {path} | num_records={n} | min_id={min_id}, max_id={max_id} | BOS={bos_cnt}, EOS={eos_cnt}{flag_str}")

    except Exception as e:
        print(f"[ERROR] {path}: {e}")
