#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example sequence to sequence agent for ParlAI "Creating an Agent" tutorial.
http://parl.ai/static/docs/tutorial_seq2seq.html
"""
import math

from parlai.core.torch_agent import TorchAgent, Output

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from parlai.core.utils import NEAR_INF


class UnknownDropout(nn.Module):
    """With set frequency, replaces tokens with unknown token.
    This layer can be used right before an embedding layer to make the model
    more robust to unknown words at test time.
    """

    def __init__(self, unknown_idx, probability):
        """Initialize layer.
        :param unknown_idx: index of unknown token, replace tokens with this
        :param probability: during training, replaces tokens with unknown token
                            at this rate.
        """
        super().__init__()
        self.unknown_idx = unknown_idx
        self.prob = probability

    def forward(self, input):
        """If training and dropout rate > 0, masks input with unknown token."""
        if self.training and self.prob > 0:
            mask = input.new(input.size()).float().uniform_(0, 1) < self.prob
            input.masked_fill_(mask, self.unknown_idx)
        return input

class RNNEncoder(nn.Module):
    """RNN Encoder."""

    def __init__(self, num_features, embeddingsize, hiddensize,
                 padding_idx=0, rnn_class='lstm', numlayers=2, dropout=0.1,
                 bidirectional=True, shared_lt=None, shared_rnn=None,
                 input_dropout=0, unknown_idx=None, sparse=False):
        """Initialize recurrent encoder."""
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hiddensize

        if input_dropout > 0 and unknown_idx is None:
            raise RuntimeError('input_dropout > 0 but unknown_idx not set')
        self.input_dropout = UnknownDropout(unknown_idx, input_dropout)

        if shared_lt is None:
            self.lt = nn.Embedding(num_features, embeddingsize,
                                   padding_idx=padding_idx,
                                   sparse=sparse)
        else:
            self.lt = shared_lt

        if shared_rnn is None:
            self.rnn = rnn_class(embeddingsize, hiddensize, numlayers,
                                 dropout=dropout if numlayers > 1 else 0,
                                 batch_first=True, bidirectional=bidirectional)
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def forward(self, xs):
        """Encode sequence.
        :param xs: (bsz x seqlen) LongTensor of input token indices
        :returns: encoder outputs, hidden state, attention mask
            encoder outputs are the output state at each step of the encoding.
            the hidden state is the final hidden state of the encoder.
            the attention mask is a mask of which input values are nonzero.
        """
        bsz = len(xs)

        # embed input tokens
        xs = self.input_dropout(xs)
        xes = self.dropout(self.lt(xs))
        attn_mask = xs.ne(0)
        try:
            x_lens = torch.sum(attn_mask.int(), dim=1)
            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output,
                                                    batch_first=True)
        if self.dirs > 1:
            # project to decoder dimension by taking sum of forward and back
            if isinstance(self.rnn, nn.LSTM):
                hidden = (hidden[0].view(-1, self.dirs, bsz, self.hsz).sum(1),
                          hidden[1].view(-1, self.dirs, bsz, self.hsz).sum(1))
            else:
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).sum(1)

        return encoder_output, hidden, attn_mask


class RNNDecoder(nn.Module):
    """Recurrent decoder module.
    Can be used as a standalone language model or paired with an encoder.
    """

    def __init__(self, num_features, embeddingsize, hiddensize,
                 padding_idx=0, rnn_class='lstm', numlayers=2, dropout=0.1,
                 bidir_input=False, attn_type='none', attn_time='pre',
                 attn_length=-1, sparse=False):
        """Initialize recurrent decoder."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.hsz = hiddensize
        self.esz = embeddingsize

        self.lt = nn.Embedding(num_features, embeddingsize,
                               padding_idx=padding_idx, sparse=sparse)
        self.rnn = rnn_class(embeddingsize, hiddensize, numlayers,
                             dropout=dropout if numlayers > 1 else 0,
                             batch_first=True)

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(attn_type=attn_type,
                                        hiddensize=hiddensize,
                                        embeddingsize=embeddingsize,
                                        bidirectional=bidir_input,
                                        attn_length=attn_length,
                                        attn_time=attn_time)

    def forward(self, xs, hidden=None, attn_params=None):
        """Decode from input tokens.
        :param xs:          (bsz x seqlen) LongTensor of input token indices
        :param hidden:      hidden state to feed into decoder. default (None)
                            initializes tensors using the RNN's defaults.
        :param attn_params: (optional) tuple containing attention parameters,
                            default AttentionLayer needs encoder_output states
                            and attention mask (e.g. encoder_input.ne(0))
        :returns:           output state(s), hidden state.
                            output state of the encoder. for an RNN, this is
                            (bsz, seq_len, num_directions * hiddensize).
                            hidden state will be same dimensions as input
                            hidden state. for an RNN, this is a tensor of sizes
                            (bsz, numlayers * num_directions, hiddensize).
        """
        # sequence indices => sequence embeddings
        xes = self.dropout(self.lt(xs))

        if self.attn_time == 'pre':
            # modify input vectors with attention
            xes, _attw = self.attention(xes, hidden, attn_params)

        # feed tokens into rnn
        output, new_hidden = self.rnn(xes, hidden)

        if self.attn_time == 'post':
            # modify output vectors with attention
            output, _attw = self.attention(output, new_hidden, attn_params)

        return output, new_hidden


class ZySeq2seqAgent(TorchAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model is based on Sean Robertson's `seq2seq tutorial
    <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.
    """
    
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(ZySeq2seqAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=1,
                           help='learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.001,
                           help='Report frequency of prediction during eval.')
        agent.add_argument('-zy', type=int, default=0,
                   help='Whether use zy')
        ZySeq2seqAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Initialize example seq2seq agent.

        :param opt: options dict generated by parlai.core.params:ParlaiParser
        :param shared: optional shared dict with preinitialized model params
        """
        super().__init__(opt, shared)
        RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

        self.id = 'Seq2Seq'

        #zhangying
        self.zy = opt['zy']
        if self.zy:
            self.criterion_cos = nn.CosineEmbeddingLoss()

        hsz = opt['hiddensize']
        nl = opt['numlayers']
        lookuptable = 'enc_dec'
        decoder = 'same'
        
        self.START_IDX = 1
        self.NULL_IDX = 0
        #self.register_buffer('START', torch.LongTensor([self.start_idx]))
        self.longest_label = 50

        rnn_class = RNN_OPTS['lstm']
        self.attn_type = "general"
        self.decoder = RNNDecoder(
            len(self.dict), hsz, hsz,
            padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            numlayers=nl,
            attn_type=self.attn_type, attn_length=48,
            attn_time="post",
            bidir_input=True)

        shared_lt = (self.decoder.lt  # share embeddings between rnns
                     if lookuptable in ('enc_dec', 'all') else None)
        shared_rnn = self.decoder.rnn if decoder == 'shared' else None
        self.encoder = RNNEncoder(
            len(self.dict), hsz, hsz,
            padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            numlayers=nl, 
            bidirectional=True,
            shared_lt=shared_lt, shared_rnn=shared_rnn,
            unknown_idx=self.dict[self.dict.unk_token])

        shared_weight = (self.decoder.lt.weight  # use embeddings for projection
                         if lookuptable in ('dec_out', 'all') else None)
        
        self.output = OutputLayer(
            len(self.dict), hsz, hsz, 
            padding_idx=self.NULL_IDX)


        if self.use_cuda:  # set in parent class
            self.encoder.cuda()
            self.decoder.cuda()
            self.output.cuda()

        # set up the criterion
        self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX)

        # set up optims for each module
        lr = opt['learningrate']
        self.optims = {
            'encoder': optim.SGD(self.encoder.parameters(), lr=lr),
            'decoder': optim.SGD(self.decoder.parameters(), lr=lr),
        }

        self.START = torch.LongTensor([self.START_IDX])
        if self.use_cuda:
            self.START = self.START.cuda()

        #self.reset()

    def _encode(self, xs, prev_enc=None):
        """Encode the input or return cached encoder state."""
        if prev_enc is not None:
            return prev_enc
        else:
            return self.encoder(xs)
        
    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def _decode_forced(self, ys, encoder_states):
        """Decode with teacher forcing."""
        bsz = ys.size(0)
        seqlen = ys.size(1)

        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])

        # input to model is START + each target except the last
        y_in = ys.narrow(1, 0, seqlen - 1)
        xs = torch.cat([self._starts(bsz), y_in], 1)

        scores = []
        scores_q = []
        if self.attn_type == 'none':
            # do the whole thing in one go
            output, hidden = self.decoder(xs, hidden, attn_params)
            score, score_q = self.output(output)
            scores.append(score)
            scores_q.append(score_q)
        else:
            # need to feed in one token at a time so we can do attention
            # TODO: do we need to do this? actually shouldn't need to since we
            # don't do input feeding
            for i in range(seqlen):
                xi = xs.select(1, i).unsqueeze(1)
                output, hidden = self.decoder(xi, hidden, attn_params)
                score, score_q = self.output(output)
                scores.append(score)
                scores_q.append(score_q)

        scores = torch.cat(scores, 1)
        scores_q = torch.cat(scores_q, 1)
        return scores, scores_q

    def _decode(self, encoder_states, maxlen):
        """Decode maxlen tokens."""
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])
        bsz = encoder_states[0].size(0)

        xs = self._starts(bsz)  # input start token

        scores = []
        scores_q = []
        for _ in range(maxlen):
            # generate at most longest_label tokens
            output, hidden = self.decoder(xs, hidden, attn_params)
            score, score_q = self.output(output)
            scores.append(score)
            scores_q.append(score_q)
            xs = score.max(2)[1]  # next input is current predicted output

        scores = torch.cat(scores, 1)
        scores_q = torch.cat(scores_q, 1)
        return scores, scores_q

    def zero_grad(self):
        """Zero out optimizer."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared['encoder'] = self.encoder
        shared['decoder'] = self.decoder
        return shared

    def v2t(self, vector):
        """Convert vector to text.

        :param vector: tensor of token indices.
            1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == self.END_IDX:
                    break
                else:
                    output_tokens.append(token)
            return self.dict.vec2txt(output_tokens)
        elif vector.dim() == 2:
            return [self.v2t(vector[i]) for i in range(vector.size(0))]
        raise RuntimeError('Improper input to v2t with dimensions {}'.format(
            vector.size()))

    def vectorize(self, *args, **kwargs):
        """Call vectorize without adding start tokens to labels."""
        kwargs['add_start'] = False
        return super().vectorize(*args, **kwargs)

    def train_step(self, batch):
        """Train model to produce ys given xs.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.
        batch
            :param text_vec:
                bsz x seqlen tensor containing the parsed text data.
            :param text_lengths:
                list of length bsz containing the lengths of the text in same order as
                text_vec; necessary for pack_padded_sequence.
            :param label_vec:
                bsz x seqlen tensor containing the parsed label (one per batch row).
            :param label_lengths:
                list of length bsz containing the lengths of the labels in same order as
                label_vec.

        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        self.zero_grad()
        self.encoder.train()
        self.decoder.train() 
        
        xs, ys = batch.text_vec, batch.label_vec
        if xs is None:
            return
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))
            
        loss = 0
        # save largest seen label for later

        encoder_states = self._encode(xs, None)

        # Teacher forcing: Feed the target as the next input
        if ys is not None:
            # use teacher forcing
            scores, scores_q = self._decode_forced(ys, encoder_states)
        else:
            scores, scores_q = self._decode(encoder_states, self.longest_label)

        _max_score, predictions = scores.max(2)

        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, ys.view(-1))

        if self.zy:
            #zhangying
            q_mask = ys != 0  # BxL
            i_mask = xs != 0  # BxL
            batch_size = xs.size()[0]
            
            i_length = xs.new_tensor(batch.text_lengths).float().view(-1,1,1)  #Bx1x1
            t_length = ys.new_tensor(batch.label_lengths).float().view(-1,1,1)  #BX1X1
        
            q_mask = q_mask.unsqueeze(2).float() # BxLx1
            i_mask = i_mask.unsqueeze(2).float() # BxLx1
           
            q_batch = scores_q.transpose(1,2) # BxVocab_sizexL
            embed_inputs = self.encoder.lt(xs).transpose(1,2) #BxEmbed_sizexL

            sum_qs = torch.bmm(q_batch, q_mask).transpose(1,2) #Bx1xVocab_size  
            #sum_qs = torch.bmm(output.transpose(0,1).transpose(1,2), q_mask).transpose(1,2) #Bx1xVocab_size
            #sum_qs = self.generator_q(sum_qs)
            #sum_qs = F.softmax(sum_qs, dim=2)
            sum_is = torch.bmm(embed_inputs, i_mask).transpose(1,2)  #Bx1xEmbed_size    
    
            source_vocab = xs.new_tensor(range(len(self.dict))).view(1,-1)
            vi_embedding_matrix = self.encoder.lt(source_vocab).view(len(self.dict),-1) #Vocab_sizexEmbed_size

            multi_q_emb = torch.matmul(sum_qs, vi_embedding_matrix) #Bx1xEmbed_size
            average_q = torch.div(multi_q_emb, t_length)
            average_embed_i = torch.div(sum_is, i_length)         
            
            #zy compute cosine loss
            sets = average_embed_i.new_ones(batch_size).view(batch_size,1)
            loss_cos = self.criterion_cos(average_embed_i.squeeze(1), average_q.squeeze(1), sets)
            loss += loss_cos
        
        loss.backward()
        self.update_params()
        return Output(self.v2t(predictions))


    def eval_step(self, batch):
        """Generate a response to the input tokens.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return predicted responses (list of strings of length batchsize).
        """
        self.encoder.eval()
        self.decoder.eval()
        
        xs = batch.text_vec
        if xs is None:
            return
        bsz = xs.size(0)
        # just predict

        encoder_states = self._encode(xs, None)
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])
        bsz = encoder_states[0].size(0)

        xs = self._starts(bsz)  # input start token

        predictions = []
        done = [False for _ in range(bsz)]
        total_done = 0
        for _ in range(self.longest_label):
            # generate at most longest_label tokens
            output, hidden = self.decoder(xs, hidden, attn_params)
            score, score_q = self.output(output)
            xs = score.max(2)[1]  # next input is current predicted output
            predictions.append(xs)

            # check if we've produced the end token
            for b in range(bsz):
                if not done[b]:
                    # only add more tokens for examples that aren't done
                    if xs[b].item() == self.END_IDX:
                        # if we produced END, we're done
                        done[b] = True
                        total_done += 1
            if total_done == bsz:
                # no need to generate any more
                break
        predictions = torch.cat(predictions, 1)
        return Output(self.v2t(predictions))

class OutputLayer(nn.Module):
    """Takes in final states and returns distribution over candidates."""

    def __init__(self, num_features, embeddingsize, hiddensize, dropout=0.1,
                 numsoftmax=1, shared_weight=None, padding_idx=-1):
        """Initialize output layer.
        :param num_features:  number of candidates to rank
        :param hiddensize:    (last) dimension of the input vectors
        :param embeddingsize: (last) dimension of the candidate vectors
        :param numsoftmax:   (default 1) number of softmaxes to calculate.
                              see arxiv.org/abs/1711.03953 for more info.
                              increasing this slows down computation but can
                              add more expressivity to the embeddings.
        :param shared_weight: (num_features x esz) vector of weights to use as
                              the final linear layer's weight matrix. default
                              None starts with a new linear layer.
        :param padding_idx:   model should output a large negative number for
                              score at this index. if set to -1 (default),
                              this is disabled. if >= 0, subtracts one from
                              num_features and always outputs -1e20 at this
                              index. only used when shared_weight is not None.
                              setting this param helps protect gradient from
                              entering shared embedding matrices.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.padding_idx = padding_idx if shared_weight is not None else -1

        # embedding to scores
        if shared_weight is None:
            # just a regular linear layer
            self.e2s = nn.Linear(embeddingsize, num_features, bias=True)
            self.e2q = nn.Linear(embeddingsize, num_features, bias=True)

        else:
            # use shared weights and a bias layer instead
            if padding_idx == 0:
                num_features -= 1  # don't include padding
                shared_weight = shared_weight.narrow(0, 1, num_features)
            elif padding_idx > 0:
                raise RuntimeError('nonzero pad_idx not yet implemented')
            self.weight = Parameter(shared_weight)
            self.bias = Parameter(torch.Tensor(num_features))
            self.reset_parameters()
            self.e2s = lambda x: F.linear(x, self.weight, self.bias)

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.esz = embeddingsize
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hiddensize, numsoftmax, bias=False)
            self.latent = nn.Linear(hiddensize, numsoftmax * embeddingsize)
            self.activation = nn.Tanh()
        else:
            # rnn output to embedding
            if hiddensize != embeddingsize:
                # learn projection to correct dimensions
                self.o2e = nn.Linear(hiddensize, embeddingsize, bias=True)
            else:
                # no need for any transformation here
                self.o2e = lambda x: x

    def reset_parameters(self):
        """Reset bias param."""
        if hasattr(self, 'bias'):
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Compute scores from inputs.
        :param input: (bsz x seq_len x num_directions * hiddensize) tensor of
                       states, e.g. the output states of an RNN
        :returns: (bsz x seqlen x num_cands) scores for each candidate
        """
        # next compute scores over dictionary
        if self.numsoftmax > 1:
            bsz = input.size(0)
            seqlen = input.size(1) if input.dim() > 1 else 1

            # first compute different softmax scores based on input vec
            # hsz => numsoftmax * esz
            latent = self.latent(input)
            active = self.dropout(self.activation(latent))
            # esz => num_features
            logit = self.e2s(active.view(-1, self.esz))

            # calculate priors: distribution over which softmax scores to use
            # hsz => numsoftmax
            prior_logit = self.prior(input).view(-1, self.numsoftmax)
            # softmax over numsoftmax's
            prior = self.softmax(prior_logit)

            # now combine priors with logits
            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()
        else:
            # hsz => esz, good time for dropout
            e = self.dropout(self.o2e(input))
            # esz => num_features
            scores = self.e2s(e)
            scores_q = F.softmax(self.e2q(e), dim=-1)

        if self.padding_idx == 0:
            pad_score = scores.new(scores.size(0),
                                   scores.size(1),
                                   1).fill_(-NEAR_INF)
            scores = torch.cat([pad_score, scores], dim=-1)
            scores_q = torch.cat([pad_score, scores_q], dim=-1)

        return scores, scores_q

class AttentionLayer(nn.Module):
    """Computes attention between hidden and encoder states.
    See arxiv.org/abs/1508.04025 for more info on each attention type.
    """

    def __init__(self, attn_type, hiddensize, embeddingsize,
                 bidirectional=False, attn_length=-1, attn_time='pre'):
        """Initialize attention layer."""
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hiddensize
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = embeddingsize
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')

            # linear layer for combining applied attention weights with input
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, attn_params):
        """Compute attention over attn_params given input and hidden states.
        :param xes:         input state. will be combined with applied
                            attention.
        :param hidden:      hidden state from model. will be used to select
                            states to attend to in from the attn_params.
        :param attn_params: tuple of encoder output states and a mask showing
                            which input indices are nonzero.
        :returns: output, attn_weights
                  output is a new state of same size as input state `xes`.
                  attn_weights are the weights given to each state in the
                  encoder outputs.
        """
        if self.attention == 'none':
            # do nothing, no attention
            return xes, None

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        enc_out, attn_mask = attn_params
        bsz, seqlen, hszXnumdir = enc_out.size()
        numlayersXnumdir = last_hidden.size(1)

        if self.attention == 'local':
            # local attention weights aren't based on encoder states
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)

            # adjust state sizes to the fixed window size
            if seqlen > self.max_length:
                offset = seqlen - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
                seqlen = self.max_length
            if attn_weights.size(1) > seqlen:
                attn_weights = attn_weights.narrow(1, 0, seqlen)
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                # concat hidden state and encoder outputs
                hid = hid.expand(bsz, seqlen, numlayersXnumdir)
                h_merged = torch.cat((enc_out, hid), 2)
                # then do linear combination of them with activation
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                # dot product between hidden and encoder outputs
                if numlayersXnumdir != hszXnumdir:
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)
            elif self.attention == 'general':
                # before doing dot product, transform hidden state with linear
                # same as dot if linear is identity
                hid = self.attn(hid)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)

            # calculate activation scores, apply mask if needed
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask.masked_fill_((1 - attn_mask), -NEAR_INF)
            attn_weights = F.softmax(attn_w_premask, dim=1)

        # apply the attention weights to the encoder states
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        # concatenate the input and encoder states
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        # combine them with a linear layer and tanh activation
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))

        return output, attn_weights