import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import csv
import argparse
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer
from transformers import get_linear_schedule_with_warmup
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import time
import copy
import pickle
import torch
from torch import nn
import numpy as np
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from data_format import topiocqa_rewrite_GRF, topiocqa_GRF, topiocqa_rewrite, topiocqa_normal


class ANCE(RobertaForSequenceClassification):
    # class Pooler:   # adapt to DPR
    #     def __init__(self, pooler_output):
    #         self.pooler_output = pooler_output

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)

def save_model(model_output_path, model, query_tokenizer):
    if args.rewrite_file and args.GRF_file:
        output_dir = oj(args.model_output_path, 'mix-shuffle-{}-rewrite{}nums-GRF{}nums-best'.format(args.rewrite_shuffle, args.rewrite_nums, args.GRF_nums))
    elif args.rewrite_file:
        output_dir = oj(args.model_output_path, 'rewrite-shuffle-{}-original-{}-rewrite{}nums-best'.format(args.rewrite_shuffle, args.with_original, args.rewrite_nums))
    elif args.GRF_file:
        output_dir = oj(args.model_output_path, 'GRF-original-{}-{}nums-best'.format(args.with_original, args.GRF_nums))
    else:
        output_dir = oj(args.model_output_path, 'original-best')
    #check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Save checkpoint at {}".format(output_dir))

def cal_kd_loss(query_embs, oracle_query_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, oracle_query_embs)

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss


def train(args):
    # load the pretrained passage encoder model, but it will be frozen when training.
    # load conversational query encoder model
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    query_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)
    passage_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    #if args.n_gpu > 1:
    #    query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    #args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    if args.rewrite_file and args.GRF_file:
        train_dataset = topiocqa_rewrite_GRF(args, tokenizer)
    elif args.rewrite_file:
        train_dataset = topiocqa_rewrite(args, tokenizer)
    elif args.GRF_file:
        train_dataset = topiocqa_GRF(args, tokenizer)
    else:
        train_dataset = topiocqa_normal(args, tokenizer)

    train_loader = DataLoader(train_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))

    logger.info("train samples num = {}".format(len(train_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    best_loss = 1000
    #total_loss = 0
    for epoch in epoch_iterator:
        query_encoder.train()
        passage_encoder.eval()
        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()
            bt_concat = batch['bt_input_ids'].to(args.device) 
            bt_concat_mask = batch['bt_attention_mask'].to(args.device)
            concat_embs = query_encoder(bt_concat, bt_concat_mask)  # B * dim

            bt_pos_docs = batch['bt_pos_docs'].to(args.device)
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device)
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
            bt_oracle_query = batch['bt_oracle_labels'].to(args.device)
            bt_oracle_query_mask = batch['bt_oracle_labels_mask'].to(args.device)
            with torch.no_grad():
            # doc encoder's parameters are frozen
                pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                oracle_utt_embs = passage_encoder(bt_oracle_query, bt_oracle_query_mask).detach()
                if len(batch['bt_neg_docs']) == 0:  # only_in_batch negative
                    neg_doc_embs = None
                else:
                    #breakpoint()
                    #batch_size, neg_ratio, seq_len = bt_neg_docs.shape       
                    #bt_neg_docs = bt_neg_docs.view(batch_size * neg_ratio, seq_len)        
                    #bt_neg_docs_mask = bt_neg_docs_mask.view(batch_size * neg_ratio, seq_len)             
                    neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # (B * neg_ratio) * dim,      
                    #neg_doc_embs = None
            ranking_loss = cal_ranking_loss(concat_embs, pos_doc_embs, neg_doc_embs)
            kd_loss = cal_kd_loss(concat_embs, oracle_utt_embs)
            loss = ranking_loss + kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            query_encoder.zero_grad()
            global_step += 1
            
            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, Step Loss = {}".format(
                                epoch + 1,
                                global_step,
                                loss.item()))

            #if global_step % args.accumulation_steps == 0:
            #    torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            #    optimizer.step()
            #    scheduler.step()
            #    accumulated_loss = 0
            #    query_encoder.zero_grad()

            if best_loss > loss:
                save_model(args.model_output_path, query_encoder, tokenizer)
                best_loss = loss

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrained_encoder_path", type=str, default="checkpoints/ad-hoc-ance-msmarco")
    parser.add_argument("--original_file", type=str, default="datasets/topiocqa/topiocqa_train.json")
    parser.add_argument("--model_output_path", type=str, default="output/models")
    parser.add_argument("--oracle_file", type=str, default="") # file with oracle rewrite
    parser.add_argument("--rewrite_file", type=str, default=None)
    parser.add_argument("--GRF_file", type=str, default="") # file with generative relevant feedback
    parser.add_argument("--with_original", type=bool, default=False)
    parser.add_argument("--rewrite_shuffle", type=bool, default=False)
    parser.add_argument("--rewrite_nums", type=int, default=5)
    parser.add_argument("--GRF_nums", type=int, default=1)

    parser.add_argument("--print_steps", type=float, default=1000, help="Percent of steps per epoch to print once.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="num_train_epochs")
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--num_warmup_portion", type=float, default=0)
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=512, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)

    parser.add_argument("--loss_type", type=str, default="ranking")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train", help="To control how to organize the batch data.")

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    train(args)
