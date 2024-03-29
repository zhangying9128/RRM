## IWSLT 2014 De-En Translation

### Our trained models
As we mentioned in our paper, we run 3 trials with random seeds. You can use the following LocalJoint+RRM or Transformer+RRM models to reproduce our results.
| Model |Trial| Link|
|---|---|---|
| LocalJoint+RRM(alpha=0.3)| 1 | [download (.pt)](https://drive.google.com/file/d/1maHFGaND2SQJuBU4gtucdGW4FRKnmrnt/view?usp=share_link) | 
| LocalJoint+RRM(alpha=0.3)| 2 | [download (.pt)](https://drive.google.com/file/d/1uzzd1QHwURnS0RAA6lbGqnzH54vo-Uu9/view?usp=share_link) | 
| LocalJoint+RRM(alpha=0.3)| 3 | [download (.pt)](https://drive.google.com/file/d/1rzzoyiDkrqH32fMa0_RQ9jJo3NEZinMH/view?usp=share_link) | 
| Transformer+RRM(alpha=0.01)| 1 | [download (.pt)](https://drive.google.com/file/d/14fglU99TOXiX-3IR3YxRBqthJAwRatqJ/view?usp=share_link) | 
| Transformer+RRM(alpha=0.01)| 2 | [download (.pt)](https://drive.google.com/file/d/1xmLf3TSe5EA4bpZfawJx10pmdY74kChK/view?usp=share_link) | 
| Transformer+RRM(alpha=0.01)| 3 | [download (.pt)](https://drive.google.com/file/d/1HT78_x5NC3KvMV3riLEg0mOJuLw8LsdI/view?usp=share_link) | 


### Data Preprocessing
Download and preprocess the IWSLT'14 German to English dataset.
```sh
# Dataset download and preparation
bash prepare-iwslt14-31K.sh

# Restore the segmentated target file
prep=iwslt14.tokenized.31K.de-en
sed -r 's/(@@ )|(@@ ?$)//g' $prep/test.en > $prep/test.en.detok

# Dataset binarization:
bash preprocess.sh
```

### Our Pre-trained Word Embedding
You can also train LocalJoint+RRM by yourself. If you want to reproduce our results, please download the following pretrained word embedding, and put it into the folder [pretrained-emb](https://github.com/zhangying9128/RRM/tree/main/IWSLT/pretrained-emb) for training.
| Embedding | Link|
|---|---|
|src_word_embedding |[download](https://drive.google.com/file/d/12oxKhK8OL_t1dHhN-4a6LoqSiT4QjFvx/view?usp=share_link)|


### Training
You can train LocalJoint+RRM or Transformer+RRM with the following scripts on a GPU.
Please edit `--device-id` based on your GPU environment. And make sure the last checkpoint is used for LocalJoint+RRM to construct average checkpoint (if you want to reproduce our results).
```sh
bash train_localjoint_rrm.sh
```

```sh
bash train_transformer_rrm.sh
```

### Evaluation
You can evaluate the Repeat score of LocalJoint+RRM or Transformer+RRM on the test set with the following script.
Please edit `MODEL_PATH` and `CHECKPOINT` based on your setting.
```sh
bash eval_rrm.sh
```

If you want to evaluate RRM with Meteor and BLEU metrics, please refer to [Meteor](https://www.cs.cmu.edu/~alavie/METEOR/README.html) and [BLEU](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl).
