#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# /scratch/nsingh/At_ImageNet/effnet_b4_100_epochs_not_orig.yaml
# rn50_configs/rn50_16_epochs.yaml
# chkpt = '/mnt/SHARED/nsingh/ImageNet_Arch/model_2022-10-11 17:43:32_effnet_b4_iso_0_not_orig_1_clean/weights_45.pt'
# #--config-file $1 \
sleep 2s

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine

python3 main.py --data.num_workers=12 --data.in_memory=1 \
        --data.train_dataset=/scratch/nsingh/datasets/ffcv_imagenet_data/train_400_0.50_90.ffcv \
        --data.val_dataset=/scratch/nsingh/datasets/ffcv_imagenet_data/val_400_0.50_90.ffcv \
        --logging.folder=/scratch/nsingh/ImageNet_Arch/ --logging.log_level 2 \
    --adv.attack apgd --adv.n_iter 2 --adv.norm L2 --training.distributed 1 --training.batch_size 80 --validation.batch_size 16 --lr.lr 1e-3 --logging.save_freq 2  \
    --resolution.min_res 224 --resolution.max_res 224 --data.seed 0 --data.augmentations 1 --model.add_normalization 0\
    --model.updated 0 --model.not_original 1 --model.model_ema 1 --lr.lr_peak_epoch 5\
    --training.label_smoothing 0.1 --logging.addendum='_20ep_5_peak_L2_norm_2'\
    --dist.world_size 8 --training.distributed 1 --model.pretrained 0\
    --model.arch convnext_base --training.epochs 20 \
   