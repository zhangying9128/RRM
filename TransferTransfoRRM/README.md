## PERSONACHAT by TransferTransfo+RRM
### Install ParAI from our repository
```bash
pip install -r requirements.txt
python -m spacy download en
```

Please use our modified ParlAI.
```bash
cd ParlAI
python setup.py develop
```

### Our trained models
As we mentioned in our paper, we run 3 trials with random seeds. You can use the following TransferTransfo+RRM models to reproduce our results.
| Model |Trial| Link|
|---|---|---|
| TransferTransfo+RRM(alpha=0.3)| 1 | [download (.pt)](https://drive.google.com/file/d/1OQ1B3T8zlq6GzC-6JPD8fYI7BMjC3QrH/view?usp=sharing)|
| TransferTransfo+RRM(alpha=0.3)| 2 | [download (.pt)](https://drive.google.com/file/d/1pXNn6NzZITQ23yqq-Jq3JGL9bTHZtIq6/view?usp=sharing)|
| TransferTransfo+RRM(alpha=0.3)| 3 | [download (.pt)](https://drive.google.com/drive/folders/1AD2aFZY0cfUTLDhFMr_-JS_669mrSt0n?usp=sharing)|



## Using the training script
You can also train TransferTransfo+RRM by yourself. 
The dataset, GPT2 model, and tokenizer will be automatically download. 

### Training
You can train TransferTransfo+RRM with the following script on a GPU.
```sh
bash train_rrm.sh
```

### Evaluation
You can evaluate the Repeat score of TransferTransfo+RRM on the test set with the following script.
```
bash eval_rrm.sh
```

### Data Format
see `example_entry.py`, and the comment at the top.
