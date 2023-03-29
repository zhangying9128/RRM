# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from . import FairseqCriterion, register_criterion

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        
        #zhangying
        #use the cosine embedding loss offered by pytorch
        #we use the "sum" mechanism because reduce=True in label_smoothed_nll_loss 
        self.criterion_cos = nn.CosineEmbeddingLoss(reduction='sum')
        self.RRM = args.RRM
        self.RRM_scale = float(args.RRM_scale)
       
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True) #B, L, V

        lprobs = lprobs.view(-1, lprobs.size(-1)) #B*L , V
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        batch_size, max_len = sample["target"].size()

        #zhangying
        #combine RRM loss with the cross entropy loss
        if self.RRM:
            loss += self.RRM_scale * self.emb_cosine_loss(net_output[2], sample, model)


        return loss, nll_loss

    #zhangying
    #this function computes the RRM loss
    def emb_cosine_loss(self, logits, sample, model):
        #sample['net_input']['src_tokens'] batch_size x input_len
        #q_logits batch_size x target_len x vocab_size
        #target batch_size x target_len
        X = sample['net_input']['src_tokens']
        batch_size = sample['target'].size(0)
        
        #zhangying
        #We use mask to skip <pad> tokens
        X_mask = X != self.padding_idx #BXL
        X_mask = X_mask.unsqueeze(2).float() # BxLx1
        q_mask = sample['target'] != self.padding_idx #BXL
        q_mask = q_mask.unsqueeze(2).float() # BxLx1
       
        #zhangying
        #Referring to our Eq. (4)
        embed_X = model.encoder.embed_tokens(X).transpose(1,2) #BxEmbed_sizexL
        x_tilde = torch.bmm(embed_X, X_mask).transpose(1,2)  #Bx1xEmbed_size

        #zhangying
        #Referring to our Eq. (5)
        q = F.softmax(logits.transpose(1,2), dim=1) # BxVocab_sizexL, 
        q = torch.bmm(q, q_mask).transpose(1,2) #Bx1xVocab_size  
        q_tilde = torch.matmul(q, model.encoder.embed_tokens.weight) #Bx1xEmbed_size


        #zhangying
        #Referring to "-cos(x, q)" in our Eq. (7)
        #Note that Pytorch utilized "1-cos(x1,x2)" to measure the cosine embedding loss, which is different from our definition.
        #https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
        #However, the additional constant "1" here would not influence the gradient. 
        sets = x_tilde.new_ones(batch_size).view(batch_size)
        loss_cos = self.criterion_cos(x_tilde.squeeze(1), q_tilde.squeeze(1), sets)
        return loss_cos
        
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
