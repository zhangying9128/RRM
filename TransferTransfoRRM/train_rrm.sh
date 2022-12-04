#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py \
        --device cuda \
        --model_name TransferTransfo_RRM_0.3_full \
        --RRM --RRM_scale 0.3 --RRM_method 'full' \
        --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 \
        --num_candidates=4 --personality_permutations=2 --train_batch_size=2 --valid_batch_size=2 --seed 1