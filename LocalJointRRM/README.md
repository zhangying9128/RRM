## IWSLT 2014 De-En Translation by LocalJoint+RRM
### Install fairseq from our repository
Please use our modified fairseq.
```sh
cd RRM/LocalJoint_RRM/fairseq/
pip install --editable .
cd ..
```

### Our trained models
As we mentioned in our paper, we run 3 trials with random seeds. You can use the following LocalJoint+RRM models to directly reproduce our results.
| Model |Trial| Link|
|---|---|---|
| LocalJoint+RRM(alpha=0.3)| 1 | [download (.pt)](https://drive.google.com/file/d/1maHFGaND2SQJuBU4gtucdGW4FRKnmrnt/view?usp=share_link) | 
| LocalJoint+RRM(alpha=0.3)| 2 | [download (.pt)](https://drive.google.com/file/d/1uzzd1QHwURnS0RAA6lbGqnzH54vo-Uu9/view?usp=share_link) | 
| LocalJoint+RRM(alpha=0.3)| 3 | [download (.pt)](https://drive.google.com/file/d/1rzzoyiDkrqH32fMa0_RQ9jJo3NEZinMH/view?usp=share_link) | 



### Data Preprocessing
Download and preprocess the IWSLT'14 German to English dataset.
```sh
# Dataset download and preparation
bash prepare-iwslt14-31K.sh

# Dataset binarization:
bash preprocess.sh
```

### Our Pre-trained Word Embedding
You can also train LocalJoint+RRM by yourself. If you want to reproduce our results, please download the following pretrained word embedding, and put it into the folder [pretrained-emb](https://github.com/zhangying9128/RRM/tree/main/LocalJointRRM/pretrained-emb) for training.
| Embedding | Link|
|---|---|
|src_word_embedding |[download](https://drive.google.com/file/d/12oxKhK8OL_t1dHhN-4a6LoqSiT4QjFvx/view?usp=share_link)|


### Training
You can train LocalJoint+RRM with the following script on a GPU.
```sh
bash train_rrm.sh
```

### Evaluation
You can evaluate the Repeat score of LocalJoint+RRM on the test set with the following script.
```
bash eval_rrm.sh
```

If you want to evaluate RRM with Meteor and BLEU metrics, please refer to [Meteor](https://www.cs.cmu.edu/~alavie/METEOR/README.html) and [BLEU](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl).
