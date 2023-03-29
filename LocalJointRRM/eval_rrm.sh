#!/bin/bash
MODEL_PATH=experiments_iwslt/checkpoints_RRM/
CHECKPOINT=checkpoint_last10_avg.pt

for GEN_SUBSET in test ; do
CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/iwslt14.joined-dictionary.de-en --user-dir models \
    --path $MODEL_PATH/$CHECKPOINT \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 1.7 --gen-subset $GEN_SUBSET > $MODEL_PATH/${GEN_SUBSET}_eval_log.txt

python scripts/extract_translation_from_results.py \
    --result-file $MODEL_PATH/${GEN_SUBSET}_eval_log.txt \
    --translation-file $MODEL_PATH/${GEN_SUBSET}_translation.txt

python scripts/evaluate_repeat.py \
    --output-file $MODEL_PATH/${GEN_SUBSET}_translation.txt \
    --reference-file iwslt14.tokenized.31K.de-en/${GEN_SUBSET}.en.detok \
    --index-file iwslt14.tokenized.31K.de-en.index.txt

done

