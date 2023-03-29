#!/bin/bash
TEXT=wmt14_en_de
fairseq-preprocess --joined-dictionary --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14.joined-dictionary.en-de --thresholdtgt 0 --thresholdsrc 0 \
  --workers 20