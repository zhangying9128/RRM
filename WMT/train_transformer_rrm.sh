#!/usr/bin/env bash
SAVE=experiments_wmt/Transformer_RRM/
fairseq-train \
    data-bin/wmt14.joined-dictionary.en-de \
    --source-lang en --target-lang de \
    --seed 1 \
    --distributed-world-size 1 --device-id 0 \
    --arch transformer_wmt_en_de \
    --clip-norm 0 \
    --optimizer adam --lr 0.0007 --max-tokens 16384 --weight-decay 0.0001 --update-freq 2 \
    --lr-scheduler inverse_sqrt --max-update 95750 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --adam-eps '1e-09' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --RRM --RRM-scale 0.3 \
    --share-all-embeddings \
    --dropout 0.1 \
    --save-dir $SAVE \
    --no-progress-bar --log-interval 500 \
    --no-last-checkpoints --no-epoch-checkpoints > $SAVE/log.txt
