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

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node 1 run_pretrain.py --batch_size 128 --model MIM_vit_base_patch16 \
# --hog_nbins 9 --mask_ratio 0.75 \
# --epochs 400 --warmup_epochs 10 --blr 1e-3 --weight_decay 0.05 \
# --data_path "/public_bme/data/reflacx-1.0.0/" --output_dir "./output_dir/"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node 1 main_attmask.py --batch_size_per_gpu 128 \
--norm_last_layer False --momentum_teacher 0.996 --num_workers 4 --eval_every 20 \
--arch vit_base --teacher_temp 0.07 --warmup_teacher_temp_epochs 10 --epochs 400 \
--shared_head True --out_dim 8192 --local_crops_number 10 --global_crops_scale 0.25 1 \
--local_crops_scale 0.05 0.25 --pred_ratio 0.3 --pred_ratio_var 0.2 --masking_prob 0.5 \
--pred_shape attmask_high \
--subset -1 --data_path "/public_bme/data/reflacx-1.0.0/" --output_dir "./output_dir/"


# "/public_bme/data/reflacx-1.0.0/"