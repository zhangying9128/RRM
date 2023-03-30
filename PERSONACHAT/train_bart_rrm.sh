#!/bin/bash
SAVE=experiments_personachat/BART_RRM/
CUDA_VISIBLE_DEVICES=0 python train.py \
        --device cuda \
        --model_checkpoint facebook/bart-base \
        --model_name $SAVE \
        --RRM --RRM_scale 0.3 --RRM_method 'full' --SEQ2SEQ \
        --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=1 \
        --num_candidates=1 --personality_permutations=2 --train_batch_size=8 --valid_batch_size=8 --seed 1