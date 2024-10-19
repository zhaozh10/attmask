#!/usr/bin/env bash
#SBATCH -p bme_gpu2
#SBATCH --job-name=localMIM
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 5-00:00:00

set -x

source activate hpm
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node 1 run_pretrain.py --batch_size 128 --model MIM_vit_base_patch16 \
--hog_nbins 9 --mask_ratio 0.75 \
--epochs 400 --warmup_epochs 10 --blr 1e-3 --weight_decay 0.05 \
--data_path "/public_bme/data/reflacx-1.0.0/" --output_dir "./output_dir/"