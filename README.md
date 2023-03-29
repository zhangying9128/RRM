# Repetition Reduction Module (RRM)
This repository contains the source code for paper [Generic Mechanism for Reducing Repetitions in Encoder-Decoder Models](https://aclanthology.org/2021.ranlp-1.180).

## Getting Started
### Requirements
* [PyTorch](http://pytorch.org/) version == 1.9.1
* Python version >= 3.6
* omegaconf version >= 2.0.6

### Clone this repository 
```sh
git clone https://github.com/zhangying9128/RRM.git
```

### Install fairseq from our repository
Please use our modified fairseq.
```sh
cd RRM/fairseq/
pip install --editable .
cd ..
```

### Reproduce RRM
We used the source code of [Fonollosa et al., (2019)](https://github.com/jarfo/joint) to preprocess the IWSLT14 De-En dataset and train LocalJoint or Transformer with RRM.
Please check [IWSLT](https://github.com/zhangying9128/RRM/tree/main/IWSLT) for more details.

We used the source code of [Wolf et al. (2019)](https://github.com/huggingface/transfer-learning-conv-ai) to preprocess the PERSONACHAT dataset and train TransferTransfo with RRM.
Please check [TransferTransfoRRM](https://github.com/zhangying9128/RRM/tree/main/TransferTransfoRRM) for more details.

## Citation:
Please cite as:
```bibtex
@inproceedings{zhang-etal-2021-generic,
    title = "Generic Mechanism for Reducing Repetitions in Encoder-Decoder Models",
    author = "Zhang, Ying  and
      Kamigaito, Hidetaka  and
      Aoki, Tatsuya  and
      Takamura, Hiroya  and
      Okumura, Manabu",
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)",
    month = sep,
    year = "2021",
    address = "Held Online",
    publisher = "INCOMA Ltd.",
    url = "https://aclanthology.org/2021.ranlp-1.180",
    pages = "1606--1615",
    abstract = "Encoder-decoder models have been commonly used for many tasks such as machine translation and response generation. As previous research reported, these models suffer from generating redundant repetition. In this research, we propose a new mechanism for encoder-decoder models that estimates the semantic difference of a source sentence before and after being fed into the encoder-decoder model to capture the consistency between two sides. This mechanism helps reduce repeatedly generated tokens for a variety of tasks. Evaluation results on publicly available machine translation and response generation datasets demonstrate the effectiveness of our proposal.",
}
```
