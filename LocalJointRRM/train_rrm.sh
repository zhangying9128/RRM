#!/bin/bash
SAVE=experiments_iwslt/checkpoints_RRM/
mkdir -p $SAVE
python fairseq/train.py data-bin/iwslt14.joined-dictionary.de-en \
  --source-lang de --target-lang en \
  --seed 1 \
  --share-all-embeddings \
  --distributed-world-size 1 --device-id 3 \
  --user-dir models \
  --arch local_joint_attention_iwslt_de_en \
  --clip-norm 0 \
  --optimizer adam --lr 0.001  --max-tokens 4000 --min-lr '1e-09' --weight-decay 0.0001 \
  --lr-scheduler inverse_sqrt --max-update 85000 --warmup-updates 4000 --warmup-init-lr '1e-07'  --adam-betas '(0.9, 0.98)' --adam-eps '1e-09' \
  --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
  --keep-last-epochs 10 \
  --RRM --RRM-scale 0.3 \
  --save-dir $SAVE \
  --encoder-embed-path pretrained-emb/src_embeddings.txt \
  --no-progress-bar --log-interval 100 > $SAVE/log.txt

python scripts/average_checkpoints.py --inputs $SAVE \
    --num-epoch-checkpoints 10 --output "${SAVE}/checkpoint_last10_avg.pt"