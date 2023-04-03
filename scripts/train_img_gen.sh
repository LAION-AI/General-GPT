#!/bin/bash

DIR_PATH="$( dirname -- "$( readlink -f -- "$0"; )"; )"

# GPT-2 to Dalle-2 Image Generation Training
python $DIR_PATH/../image_generation/train_gpt-2_dalle-2.py \
    --models_dir $DIR_PATH/../models/ \
    --dataset "laion/laion-coco" \
    --dataset_split "train" \
    --epochs 3 \
    --lr 0.0001\
    --batch_size 64 \
    --warm_steps 5000 \
    --save_dir $DIR_PATH/../pretrained/ \