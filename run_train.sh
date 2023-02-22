#!/bin/bash

export CUDA_VISIBLE_DEVICES='DEVICE_IDS separated by comma'

sleep 2s

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine

python3 main.py --data.num_workers=12 --data.in_memory=1 \
        --data.train_dataset=path-to-imagenet-train-set \
        --data.val_dataset=path-to-imagenet-val-set \
        --logging.folder=path-to-logging-folder --logging.log_level 2 \
    --adv.attack apgd --adv.n_iter 2 --adv.norm Linf --training.distributed 1 --training.batch_size 80 --validation.batch_size 16 --lr.lr 1e-3 --logging.save_freq 2  \
    --resolution.min_res 224 --resolution.max_res 224 --data.seed 0 --data.augmentations 1 --model.add_normalization 0\
    --model.updated 0 --model.not_original 1 --model.model_ema 1 --lr.lr_peak_epoch 20\
    --training.label_smoothing 0.1 --logging.addendum='additional_text_appended_to_save_folder_name'\
    --dist.world_size '# of GPUS' --training.distributed 1 --model.pretrained 0\
    --model.arch convnext_base --training.epochs 300 \
   
