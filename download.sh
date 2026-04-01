i=${1:-1}

huggingface-cli download mlfoundations/dclm-baseline-1.0 \
  --repo-type dataset \
  --include "global-shard_0${i}_of_10/**" \
  --local-dir dclm_subset \
  --local-dir-use-symlinks False