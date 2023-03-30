#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate pre-trained model trained for f1 metric.
This seq2seq model was trained on convai2:self.
""" 

from parlai.core.build_data import download_models
from parlai.scripts.eval_wordstat import eval_wordstat as run_eval_wordstat, setup_args as setup_wordstat_args
from projects.convai2.build_dict import build_dict, DICT_FILE

def setup_args(parser=None):
    parser = setup_wordstat_args(parser)
    parser.set_defaults(
        #task='convai2:self', 
        #datatype='valid', 
        task='personachat:self', 
        datatype='test', 
        dict_tokenizer='split',
        external_dict=DICT_FILE,
        #dump_predictions_path='baseline/none_seq2seq_model2_prediction_test.txt',
    )
    return parser

def eval_wordstat(opt):
    return run_eval_wordstat(opt)
    
if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args(print_args=False)
    '''if (opt.get('model_file', '')
            .find('convai2/seq2seq/convai2_self_seq2seq_model') != -1):
        opt['model_type'] = 'seq2seq'
        fnames = ['convai2_self_seq2seq_model.tgz',
                  'convai2_self_seq2seq_model.dict',
                  'convai2_self_seq2seq_model.opt']
        download_models(opt, fnames, 'convai2', version='v3.0')'''
    #build_dict()  # make sure true dictionary is built
    eval_wordstat(opt)
