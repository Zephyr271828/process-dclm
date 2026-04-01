import glob
import json
from dclm_paths import DATASET_ROOT, INDEX_PATH

K = 32

all_paths = sorted(
    str(path.relative_to(DATASET_ROOT))
    for path in DATASET_ROOT.glob("global-shard_*/*/*.jsonl.zst")
    if path.is_file()
)

index = {i: all_paths[i * K:(i + 1) * K] for i in range(len(all_paths) // K)}

with INDEX_PATH.open("w") as f:
    json.dump(index, f, indent=2)
    
print(len(index))
