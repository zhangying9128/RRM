#!/bin/bash
TEXT=iwslt14.tokenized.31K.de-en
python fairseq/preprocess.py --joined-dictionary --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.joined-dictionary.de-en