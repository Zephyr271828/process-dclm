i=${1:-1}

hf download mlfoundations/dclm-baseline-1.0 \
  --repo-type dataset \
  --include "global-shard_0${i}_of_10/**" \
  --local-dir dclm_subset
