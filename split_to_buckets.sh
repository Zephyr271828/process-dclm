for i in {0..346}; do
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=split_to_buckets_$i
#SBATCH --output=logs/split_to_buckets_$i.out
#SBATCH --error=logs/split_to_buckets_$i.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fms

python /n/fs/vision-mix/yx1168/pruning/datasets/dclm/scripts/split_to_buckets.py --i $i
EOF

done