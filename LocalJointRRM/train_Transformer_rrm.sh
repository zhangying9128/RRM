#!/usr/bin/env bash
SAVE=experiments_iwslt/Transformer_RRM/
mkdir -p $SAVE
fairseq-train \
    data-bin/iwslt14.joined-dictionary.de-en/ \
    --source-lang de --target-lang en \
    --seed 1 \
    --share-decoder-input-output-embed \
    --distributed-world-size 1 --device-id 0 \
    --arch transformer_iwslt_de_en \
    --clip-norm 0 \
    --optimizer adam --lr 5e-4 --max-tokens 8192 --weight-decay 0.0001 \
    --lr-scheduler inverse_sqrt --max-update 42500 --warmup-updates 2000  --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' --adam-eps '1e-09' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --RRM --RRM-scale 0.01 \
    --dropout 0.3  \
    --save-dir $SAVE \
    --no-progress-bar --log-interval 100 \
    --no-last-checkpoints --no-epoch-checkpoints > $SAVE/log.txt