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
As we mentioned in our paper, we run 3 trials with random seeds. You can use the following TransferTransfo+RRM or BART+RRM models to reproduce our results.
| Model |Trial| Link|
|---|---|---|
| TransferTransfo+RRM(alpha=0.3)| 1 | [download (.pt)](https://drive.google.com/file/d/1OQ1B3T8zlq6GzC-6JPD8fYI7BMjC3QrH/view?usp=sharing)|
| TransferTransfo+RRM(alpha=0.3)| 2 | [download (.pt)](https://drive.google.com/file/d/1pXNn6NzZITQ23yqq-Jq3JGL9bTHZtIq6/view?usp=sharing)|
| TransferTransfo+RRM(alpha=0.3)| 3 | [download (.pt)](https://drive.google.com/drive/folders/1AD2aFZY0cfUTLDhFMr_-JS_669mrSt0n?usp=sharing)|
| BART+RRM(alpha=0.3)| 1 | [download (.pt)](https://drive.google.com/file/d/1W_oleVMUbLSJVQIJKNv3eT39Jvk4uG44/view?usp=share_link) | 
| BART+RRM(alpha=0.3)| 2 | [download (.pt)](https://drive.google.com/file/d/1eOh2OC0zudm-zANM7d24XCuhYZ2bYcEb/view?usp=share_link) | 
| BART+RRM(alpha=0.3)| 3 | [download (.pt)](https://drive.google.com/file/d/1Xu_huddKFcnaq7nLM1Ed_nfJWDhAFmPr/view?usp=share_link) | 

## Using the training script
You can also train TransferTransfo+RRM or BART+RRM by yourself. 
The dataset, GPT2 model, BART model, and tokenizer will be automatically downloaded. 

### Training
You can train TransferTransfo+RRM or BART+RRM with the following scripts on a GPU.
```sh
bash train_transfertransfo_rrm.sh
```

```sh
bash train_bart_rrm.sh
```

### Evaluation
You can evaluate the Repeat score of TransferTransfo+RRM or BART+RRM on the test set with the following scripts.
```sh
bash eval_transfertransfo_rrm.sh
```

```sh
bash eval_bart_rrm.sh
```
### Data Format
see `example_entry.py`, and its comment at the top.
