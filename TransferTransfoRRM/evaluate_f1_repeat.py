# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
from collections import Counter, defaultdict
from nltk.util import ngrams
from argparse import ArgumentParser

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s, set='official'):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def tokeni(text):
        text = re.sub(re_punc, lambda x: " "+x.group(0), text)
        return text

    def replace_simple(text):
        text = re.sub("'s", 'is', text)
        text = re.sub("'ve", 'have', text)
        text = re.sub("'m", 'am', text)
        text = re.sub("'ll", 'will', text)
        text = re.sub("don 't", 'do not', text)
        text = re.sub("didn 't", 'did not', text)
        text = re.sub("'re", 'are', text)
        text = re.sub("'d", 'would', text)
        return text	

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()
    if set == 'official':
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    else:
        return replace_simple(tokeni(lower(s)))

def compute_repeat(outputs):
    gram = 5
    gram_repeat = [0 for g in range(1, gram+1)]

    for i, output in enumerate(outputs): 
        for g in range(1, gram+1):
            sentence = []
            for tokens in ngrams(output, g):
                sentence += [' '.join(tokens)]
            gram_repeat[g-1] += eval_pair_repeat(sentence)

    gram_repeat = [value / len(outputs) for value in gram_repeat]
    print(gram_repeat)
    return gram_repeat

def eval_pair_repeat(output):
    output = Counter(output)
    Repeat = 0
    for token, count in output.items():
        if 'rrmeos' in token:
            continue
        if count > 1:
            Repeat += max(0, count-1)
    return Repeat


def _prec_recall_f1_score(pred_items, gold_items):
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    
def compute_macro_f1(d1, d2):
    macro_f1 = 0
    for a, b in zip(d1, d2):
        macro_f1 +=  _prec_recall_f1_score(a, b)
    macro_f1 = macro_f1 / len(d1)
    return macro_f1


def run():
    parser = ArgumentParser()
    parser.add_argument("--output-file", type=str, default="", help="")
    parser.add_argument("--reference-file", type=str, default="", help="")
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        outputs = f.readlines()
    with open(args.reference_file, 'r') as f:
        references = f.readlines()
    
    predict_sentences = []
    predict_dialogs = []
    history = []
    references = [line.strip() for line in references]
    for line in outputs:
        line = line.strip()
        if line.startswith('CONTEXT: your persona'):
            if history != []:
                predict_dialogs.append(' rrmeos '.join(history))
                history = []
        elif line.startswith('PREDICTION:'):
            predict_sentences.append(line.split('PREDICTION:')[1])
            history.append(line.split('PREDICTION:')[1])
    predict_dialogs.append(' rrmeos '.join(history))

    print('------ evaluation -----')
    predict_sentences = [normalize_answer(line, 'official').split() for line in predict_sentences]
    predict_dialogs = [normalize_answer(line, 'official').split() for line in predict_dialogs]
    references = [normalize_answer(line, 'official').split() for line in references]
    macro_f1 = compute_macro_f1(predict_sentences, references)
    print('macro_f1', macro_f1, '\n')
    print('1~5 gram Repeat (sentence-level) score')
    s_repeat = compute_repeat(predict_sentences) #sentence level
    print('1~5 gram Repeat (document-level) score')
    d_repeat = compute_repeat(predict_dialogs) #dialog level

if __name__ == "__main__":
    run()
