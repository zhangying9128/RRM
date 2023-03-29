## WMT 2014 En-De Translation

### Our trained models
As we mentioned in our paper, we run 3 trials with random seeds. You can use the following Transformer+RRM models to reproduce our results.
| Model |Trial| Link|
|---|---|---|
| Transformer+RRM(alpha=0.3)| 1 | [download (.pt)](https://drive.google.com/file/d/1muX7lsbbdIhj2PEUYs6nOBi9j9JAtYe0/view?usp=share_link) | 
| Transformer+RRM(alpha=0.3)| 2 | [download (.pt)](https://drive.google.com/file/d/1jYit32XfvW5G-yAFqg1WOnnr_lixofNs/view?usp=share_link) | 
| Transformer+RRM(alpha=0.3)| 3 | [download (.pt)](https://drive.google.com/file/d/1QQzeo0ZoOmlpXlBGa1r-kljdOr0bcd8m/view?usp=share_link) | 


### Data Preprocessing
Download and preprocess the WMT'14 English to German dataset.
```sh
# Dataset download and preparation
bash prepare-wmt14en2de.sh --icml17

# Restore the segmentated target file
prep=wmt14_en_de
sed -r 's/(@@ )|(@@ ?$)//g' $prep/test.de > $prep/test.de.detok

# Dataset binarization:
bash preprocess.sh
```

### Training
You can train Transformer+RRM with the following script on a GPU.
Please edit '--device-id ' based on your GPU environment.
```sh
bash train_transformer_rrm.sh
```

### Evaluation
You can evaluate the Repeat score of Transformer+RRM on the test set with the following script.
Please edit 'MODEL_PATH' and 'CHECKPOINT' based on your setting.
```sh
bash eval_rrm.sh
```

If you want to evaluate RRM with Meteor and BLEU metrics, please refer to [Meteor](https://www.cs.cmu.edu/~alavie/METEOR/README.html) and [BLEU](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl).
