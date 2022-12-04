#!/bin/bash
SAVE=runs/all_0.3/seed1/
#compute perplexity
#CUDA_VISIBLE_DEVICES=0 python convai_evaluation.py --eval_type ppl --model_checkpoint $SAVE --top_p 0 --datatype test

#Generate output by beam search
OUTPUT=test_beam_output.txt
CUDA_VISIBLE_DEVICES=0 python convai_evaluation.py --eval_type wordstat --model_checkpoint $SAVE --top_p 0 --datatype test --save_name $OUTPUT

python evaluate_f1_repeat.py --output-file $SAVE/$OUTPUT --reference-file test_none_original_no_cand.txt


#Generate output by beam search without using history information
#OUTPUT=test_beam_output_wohistory.txt
#CUDA_VISIBLE_DEVICES=0 python convai_evaluation.py --eval_type wordstat --model_checkpoint $SAVE --top_p 0 --max_history 0 --datatype test --save_name $OUTPUT

#python evaluate_f1_repeat.py --output-file $SAVE/$OUTPUT --reference-file test_none_original_no_cand.txt

#Generate output by greedy decoding
#OUTPUT=test_greedy_output.txt
#CUDA_VISIBLE_DEVICES=0 python convai_evaluation.py --eval_type wordstat --model_checkpoint $SAVE --inference greedy --datatype test --save_name $OUTPUT

#python evaluate_f1_repeat.py --output-file $SAVE/$OUTPUT --reference-file test_none_original_no_cand.txt
