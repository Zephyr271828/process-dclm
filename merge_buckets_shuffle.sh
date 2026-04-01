for i in {0..127}; do
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=merge_buckets_shuffle_$i
#SBATCH --output=logs/merge_buckets_shuffle_$i.out
#SBATCH --error=logs/merge_buckets_shuffle_$i.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate fms

python /n/fs/vision-mix/yx1168/pruning/datasets/dclm/scripts/merge_buckets_shuffle.py --i $i --delete

python /n/fs/vision-mix/yx1168/pruning/datasets/dclm/scripts/upload.py --i $i
EOF

done