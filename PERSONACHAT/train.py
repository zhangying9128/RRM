# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
import random
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from utils import get_dataset, make_logdir

#zhangying
from transformers import BartTokenizer, BartForConditionalGeneration



logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0, SEQ2SEQ=False):
    if not SEQ2SEQ:
        PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids", "persona_input_ids", "history_input_ids", "query_input_ids"]
    else:
        PADDED_INPUTS = ["input_ids", "target_ids", "lm_labels", "input_type_ids", "target_type_ids"]

    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer, SEQ2SEQ):
    if not SEQ2SEQ:
        ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
    else:
        ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['madeupword0000', 'madeupword0001']}
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True, SEQ2SEQ=False):
    if not SEQ2SEQ:
        SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
    else:
        SPECIAL_TOKENS = ["<s>", "</s>", "madeupword0000", "madeupword0001", "<pad>"]
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    if not SEQ2SEQ:
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        #zhangying
        instance["persona_input_ids"] = [bos] + list(chain(*persona))
        instance["history_input_ids"] = list(chain(*sequence[1:-2]))
        instance["query_input_ids"] = sequence[-2] #reply + ([eos] if with_eos else [])

    else:
        #seq2seq
        instance["input_ids"] = list(chain(*sequence[:-1])) +[eos]
        instance["target_ids"] = sequence[-1][:-1]
        instance["input_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[:-1]) for _ in s]
        instance["target_type_ids"] = [speaker1 for _ in instance["target_ids"]]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = [-1] * (len(sequence[-1]) - 1)
        if lm_labels:
            instance["lm_labels"] = sequence[-1][1:]
    return instance


def get_data_loaders(args, tokenizer):
    if not args.SEQ2SEQ:
        MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids", "persona_input_ids", "history_input_ids", "query_input_ids"]
    else:
        #seq2seq
        MODEL_INPUTS = ["input_ids", "target_ids", "mc_token_ids", "lm_labels", "mc_labels", "input_type_ids", "target_type_ids"]

    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.SEQ2SEQ:
            num_candidates = 1
        else:
            if args.num_candidates > 0 and dataset_name == 'train':
                num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels, SEQ2SEQ=args.SEQ2SEQ)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids("<pad>"), SEQ2SEQ=args.SEQ2SEQ)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                if not args.SEQ2SEQ or datasets[dataset_name]["n_candidates"] > 1:
                    tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))

    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument('--RRM', action='store_true', help='use rrm')
    parser.add_argument('--RRM_scale', default=0, type=float, help='referring to our scaling factor alpha in Eq. (3)')
    parser.add_argument('--RRM_method', default='full', choices=['full', 'divide', 'part'], type=str, help='referring to our setting, full, divide, and part in Section 4.3')   
    parser.add_argument('--SEQ2SEQ', action='store_true', help='use rrm')    
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument('--model_name', default='model', help='model name')

    
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    if not args.SEQ2SEQ:
        tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
        model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
        model = model_class.from_pretrained(args.model_checkpoint)

    else:
        #config = TransformerConfig()
        #model = TransformerModel(config)
        tokenizer_class = BartTokenizer
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
        model_class = BartForConditionalGeneration
        model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    #zhangying
    model.config.RRM = args.RRM
    model.config.RRM_scale = args.RRM_scale
    model.config.RRM_method = args.RRM_method      
    model.config.seed = args.seed  
    
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer, args.SEQ2SEQ)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)
    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if not args.SEQ2SEQ:
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, persona_input_ids, history_input_ids, query_input_ids = batch
        else:
            #seq2seq
            input_ids, target_ids, mc_token_ids, lm_labels, mc_labels, input_type_ids, target_type_ids = batch

        #zhangying
        loss_rrm = 0.0
        if not args.SEQ2SEQ:
            (lm_loss), (mc_loss), *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
                mc_labels=mc_labels, lm_labels=lm_labels#, position_ids=position_ids
            )
        else:
            #seq2seq
            (lm_loss), (mc_loss), *_ = model(
                input_ids, input_ids!=tokenizer.convert_tokens_to_ids("<pad>"), target_ids, target_ids!=tokenizer.convert_tokens_to_ids("<pad>"), labels=lm_labels,
                return_dict=False,
            )

        #start compute RRM loss
        if args.RRM:
            #    **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``
            #    **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``
            #    **hidden_states = model.hidden_states #batch_size x num_choices x sequence_length x hidden_size

            #    input_ids = inputs + [pad] + targets + [eos]
            
            batch_size = len(input_ids)
            if not args.SEQ2SEQ:
                if args.RRM_method == "part":
                    inputs_lists = [query_input_ids]
                elif args.RRM_method == "full":
                    inputs_lists = [torch.cat((persona_input_ids, history_input_ids, query_input_ids), 2)]
                elif args.RRM_method == "divide":
                    inputs_lists = [persona_input_ids, history_input_ids, query_input_ids]   
            else: 
                inputs_lists = [input_ids]

            count_X = input_ids.new_ones((batch_size,1), dtype=torch.float) * len(inputs_lists)
            if not args.SEQ2SEQ:
                hidden_states = model.hidden_states[:, -1, :, :].clone()
            else:
                hidden_states = model.hidden_states.clone()
            loss_rrm_divide = input_ids.new_zeros((batch_size,1), dtype=torch.float)
            for part_i, RRM_input in enumerate(inputs_lists):
                if not args.SEQ2SEQ:
                    X = RRM_input[:,-1,:] #B x L
                else:
                    X = RRM_input #B x L

                #We use mask to skip <pad> and -1 tokens
                X_mask = X != tokenizer.convert_tokens_to_ids("<pad>") # B x L 
                X_mask = X_mask.unsqueeze(2).float() # B x L x 1
                X_length = X_mask.sum(dim=1).view(-1) #B
                if not args.SEQ2SEQ:
                    q_mask = lm_labels[:,-1,1:] != -1 # B x L
                    q_mask = torch.cat((q_mask, q_mask.new_zeros((batch_size,1))), dim=1)
                else:
                    q_mask = lm_labels != -1 #B x L
                q_mask = q_mask.unsqueeze(2).float() # Bx xLx1            
                
                #Referring to our Eq. (4)
                if not args.SEQ2SEQ:
                    embed_X = model.transformer.tokens_embed(inputs).transpose(1,2) #B x Embed_size x L
                else:
                    embed_X = model.model.shared(inputs).transpose(1,2) #B x Embed_size x L
                x_tilde = torch.matmul(embed_X, X_mask).transpose(1,2).squeeze(1)  #B x Embed_size

                #make sure length of X > 0
                ummask_batch_id = []
                for batch_id in range(batch_size):
                    if X_length[batch_id] == 0:
                        count_X[batch_id][0] -= 1
                    else:
                        ummask_batch_id.append(batch_id)
                ummask_batch_id = torch.tensor(ummask_batch_id, device=args.device)

                #Referring to our Eq. (5)
                if args.RRM_method == "divide":
                    if part_i == 0:
                        generator_q = model.generator_q_part1
                    elif part_i == 1:
                        generator_q = model.generator_q_part2    
                    elif part_i == 2:
                        generator_q = model.generator_q_part3

                else:
                    generator_q = model.generator_q
                q_logits = F.linear(hidden_states, generator_q).transpose(1,2)
                q = F.softmax(q_logits, dim=1) # BxVocab_sizexL
                q = torch.matmul(q, q_mask).transpose(1,2) #Bx 1 x Vocab_size
                if not args.SEQ2SEQ:
                    q_tilde = torch.matmul(q, model.transformer.tokens_embed.weight).squeeze(1) #BxEmbed_size
                else:
                    q_tilde = torch.matmul(q, model.model.shared.weight).squeeze(1) #BxEmbed_size

                #Referring to "-cos(x, q)" in our Eq. (7)
                #Note that Pytorch utilized "1-cos(x1,x2)" to measure the cosine embedding loss, which is different from our definition.
                #https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
                #However, the additional constant "1" here would not influence the gradient. 
                sets = x_tilde.new_ones(batch_size).view(-1)
                if len(ummask_batch_id) != 0:
                    q_tilde = torch.index_select(q_tilde, 0, ummask_batch_id)
                    x_tilde = torch.index_select(x_tilde, 0, ummask_batch_id)
                    sets = torch.index_select(sets, 0, ummask_batch_id)
                    l_cos_none =  model.criterion_cos_none(x_tilde, q_tilde, sets)
                    loss_rrm_divide[ummask_batch_id] += l_cos_none.view(-1,1)               
                                   
            loss_rrm = torch.sum(torch.div(loss_rrm_divide, count_X)) / batch_size 

            if torch.isnan(loss_rrm):
                print("nan loss, exit")
                exit()

        #zhangying
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef + loss_rrm * args.RRM_scale) / args.gradient_accumulation_steps 
        #loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if not args.SEQ2SEQ:
                input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, persona_input_ids, history_input_ids, query_input_ids, persona_input_ids, history_input_ids, query_input_ids = batch
                logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            else:
                #seq2seq
                input_ids, target_ids, mc_token_ids, lm_labels, mc_labels, input_type_ids, target_type_ids = batch

            # if we dont send labels to model, it doesnt return losses
            if not args.SEQ2SEQ:
                lm_logits, mc_logits, *_ = model(
                    input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids#, position_ids=position_ids
                )
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
                return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
 
            else:
                lm_logits, _, *_ = model(
                    input_ids, input_ids!=tokenizer.convert_tokens_to_ids("<pad>"), target_ids, target_ids!=tokenizer.convert_tokens_to_ids("<pad>"),
                    return_dict=False,
                )
                lm_logits_flat_shifted = lm_logits.contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels.contiguous().view(-1)
      
                return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    if not args.SEQ2SEQ:
        metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
                   "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
        metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                        "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    else:
        metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
        metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_name)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
