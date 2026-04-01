import pdb
import glob
import json

k = 32

out_path = "/n/fs/vision-mix/yx1168/pruning/datasets/dclm/scripts/index.json"

root_dir = "/n/fs/vision-mix/yx1168/pruning/datasets/dclm/dclm_subset"

all_paths = sorted([ each for i in range(2, 6) for each in glob.glob(f"{root_dir}/global-shard_0{i}*/*/*.jsonl.zst")])

index = {i: all_paths[i * k:(i + 1) * k] for i in range(len(all_paths) // k)}

with open(out_path, "w") as f:
    json.dump(index, f, indent=2)
    
print(len(index))