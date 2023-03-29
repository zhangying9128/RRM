#!/bin/bash
MODEL_PATH=experiments_wmt/Transformer_RRM/
CHECKPOINT=checkpoint_best.pt

CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/wmt14.joined-dictionary.en-de \
    --path $MODEL_PATH/$CHECKPOINT \
    --batch-size 128 --beam 4 --remove-bpe --lenpen 0.6 --gen-subset test > $MODEL_PATH/test_eval_log.txt

python ../scripts/extract_translation_from_results.py \
    --result-file $MODEL_PATH/test_eval_log.txt \
    --translation-file $MODEL_PATH/test_translation.txt

python ../scripts/evaluate_repeat.py \
    --output-file $MODEL_PATH/test_translation.txt \
    --reference-file wmt14_en_de/test.de.detok \
    --index-file wmt14.tokenized.en-de.index.txt


