import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')
sys.path.append('.')
import time
import numpy as np
import argparse
from os.path import join as oj
from tqdm import tqdm, trange

import torch, os
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
#from tensorboardX import SummaryWriter
from transformers import get_linear_schedule_with_warmup

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from models import ANCE
from utils import check_dir_exist_or_build, set_seed, get_optimizer
from data import Topiocqa 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def save_model(model_output_path, model, query_tokenizer, optimizer, scheduler, cur_step, best_loss):
    """
    Save a full training checkpoint including:
      - Model weights
      - Tokenizer
      - Optimizer state
      - Scheduler state
      - Current step and best loss (trainer state)

    This structure allows you to resume training later.
    """

    # Create a unique directory per checkpoint (e.g., by step)
    output_dir = oj(model_output_path, f"checkpoint-step-{cur_step}")
    os.makedirs(output_dir, exist_ok=True)

    # ====== Save model and tokenizer ======

    # If model is wrapped in DistributedDataParallel, unwrap it
    model_to_save = model.module if hasattr(model, "module") else model

    # Save the model in HuggingFace format
    model_to_save.save_pretrained(output_dir)

    # Save the tokenizer as well
    query_tokenizer.save_pretrained(output_dir)

    # ====== Save optimizer and scheduler states ======

    # Optimizer state is required to resume training with momentum/Adam states
    torch.save(optimizer.state_dict(), oj(output_dir, "optimizer.pt"))

    # Scheduler state is required to resume the learning rate schedule
    torch.save(scheduler.state_dict(), oj(output_dir, "scheduler.pt"))

    # ====== Save training state info ======

    # This file stores metadata about training progress (e.g., step, best loss)
    torch.save({
        "step": cur_step,
        "best_loss": best_loss,
    }, oj(output_dir, "trainer_state.pt"))

    logger.info(f"Saved checkpoint at {output_dir}")


# def save_model(model_output_path, model, query_tokenizer):

#     output_dir = oj(args.model_output_path, 'test_topiocqa')
#     #check_dir_exist_or_build([output_dir])
#     os.makedirs(output_dir, exist_ok=True)
#     model_to_save = model.module if hasattr(model, 'module') else model
#     model_to_save.save_pretrained(output_dir)
#     query_tokenizer.save_pretrained(output_dir)
#     logger.info("Save checkpoint at {}".format(output_dir))


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
    #if not args.need_output:
    #    args.log_path = "./tmp"
    #if dist.get_rank() == 0:
    #    check_dir_exist_or_build([args.log_path], args.force_emptying_dir)
        #log_writer = SummaryWriter(log_dir = args.log_path)
    #else:
    #    log_writer = None

    # 1. Load query and doc encoders
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    query_tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    query_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)
    passage_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    query_encoder = DDP(query_encoder, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    dist.barrier()

    # 2. Prepare training data
    train_dataset = Topiocqa(
        args,
        query_tokenizer,
        args.training_data_file
    )

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.n_gpu > 1:
        sampler = DistributedSampler(
            train_dataset, 
            shuffle = True, 
            drop_last = True
            )
    else:
        sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.per_gpu_train_batch_size, 
        sampler=sampler, 
        collate_fn=train_dataset.get_collate_fn(args, query_tokenizer.pad_token_id),
        drop_last = True
        )

    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))

    # 3. optimizer
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)

    # 4. warmup: yuchen: warm up steps should usually be set to 0.06* total steps
    ###Learning rate
    ###^
    ###|        /\
    ###|       /  \
    ###|      /    \______
    ###|_____/             \______> training steps
    ###      ↑
    ###   warmup ending
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(args.warmup_ratio * total_training_steps), num_training_steps=total_training_steps)


    # begin to train
    logger.info("Start training...")
    logger.info("Total training samples = {}".format(len(train_dataset)))
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))
    args.log_print_steps = max(1, int(args.log_print_steps * num_steps_per_epoch))

    # if we want to resume
    if args.resume_from_checkpoint is not None:
        ckpt = args.resume_from_checkpoint

        # 1) Load model weights
        model_to_load = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        query_encoder = model_to_load.from_pretrained(ckpt).to(args.device)

        # 2) Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(ckpt, "scheduler.pt")))

        # 3) Load trainer state (global step, best loss)
        state = torch.load(os.path.join(ckpt, "trainer_state.pt"))
        cur_step = state["step"]
        best_loss = state["best_loss"]

        logger.info(f"Resumed from checkpoint {ckpt}: step={cur_step}, best_loss={best_loss}")

    else:
        cur_step = 0
        best_loss = float("inf")


    # For each Epoch
    epoch_iterator = trange(args.num_train_epochs, desc="Epoch")
    for epoch in epoch_iterator:
        # yuchen: only train query encoder, freeze doc encoder
        query_encoder.train()
        passage_encoder.eval()
        # yuchen: necessary to make the distribution of samples correct
        if args.n_gpu > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader,  desc="Step"):
            # yuchen: or optimizer.zero_grad()
            query_encoder.zero_grad()
            bt_concat = batch['bt_input_ids'].to(args.device) 
            bt_concat_mask = batch['bt_attention_mask'].to(args.device)
            concat_embs = query_encoder(bt_concat, bt_concat_mask)  # B * dim
            
            #if args.loss_type == "kd":
            #    oracle_query_encoder.eval()
            #    bt_oracle_utt = batch["bt_oracle_utt"].to(args.device)
            #    bt_oracle_utt_mask = batch["bt_oracle_utt_mask"].to(args.device)
            #    with torch.no_grad():
                # freeze oracle query encoder's parameters
            #        oracle_utt_embs = oracle_query_encoder(bt_oracle_utt, bt_oracle_utt_mask).detach()  # B * dim
            #    loss = cal_kd_loss(concat_embs, oracle_utt_embs)
            #elif args.loss_type == "ranking":
            bt_pos_docs = batch['bt_pos_docs'].to(args.device)
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device)

            # yuchen: in DPR paper, it is shown that only one BM25 hard negative can be helpful, while two do not help further.
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
                    #batch_size, neg_ratio, seq_len = bt_neg_docs.shape       
                    #bt_neg_docs = bt_neg_docs.view(batch_size * neg_ratio, seq_len)        
                    #bt_neg_docs_mask = bt_neg_docs_mask.view(batch_size * neg_ratio, seq_len)             
                    neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # (B * neg_ratio) * dim,      

            ranking_loss = cal_ranking_loss(concat_embs, pos_doc_embs, neg_doc_embs)
            kd_loss = cal_kd_loss(concat_embs, oracle_utt_embs)
            loss = ranking_loss + kd_loss
            loss.backward()
            # grad norm should be done after loss.backward and before optimizer.step
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # print info
            if dist.get_rank() == 0 and cur_step % args.log_print_steps == 0:
                logger.info("Epoch = {}, Current Step = {}, Total Step = {}, Loss = {}".format(
                    epoch,
                    cur_step,
                    total_training_steps,
                    round(loss.item(), 7))
                    )
            #if dist.get_rank() == 0:
            #    log_writer.add_scalar("train_{}_loss".format(args.loss_type), loss, cur_step)
            cur_step += 1    # avoid saving the model of the first step.
            dist.barrier()
            # Save model
            if dist.get_rank() == 0 and best_loss > loss:
                save_model(
                    args.model_output_path,
                    query_encoder,
                    query_tokenizer,
                    optimizer,
                    scheduler,
                    cur_step,
                    best_loss
                )
                
                # save_model(args.model_output_path, query_encoder, query_tokenizer) 
                logger.info("Epoch = {}, Current Step = {}, Total Step = {}, Loss = {}".format(
                                epoch,
                                cur_step,
                                total_training_steps,
                                round(loss.item(), 7))
                            )
                best_loss = loss

    logger.info("Training finish!")          
    #if dist.get_rank() == 0:   
    #    log_writer.close()
       

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local-rank', type=int, default=-1, metavar='N', help='Local process rank.')  # is useful if using python -m torch.distributed.luanch xxxxx 

    parser.add_argument('--n_gpu', type=int, default=4, help='The number of used GPU.')
    parser.add_argument("--pretrained_encoder_path", type=str, default="checkpoints/ad-hoc-ance-msmarco")
    parser.add_argument( "--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--training_data_file", type=str, default="datasets/topiocqa/topiocqa_train.json")
    parser.add_argument("--model_output_path", type=str, default="output/models")
    parser.add_argument("--oracle_file", type=str, default="") # file with oracle rewrite
    parser.add_argument("--rewrite_file", type=str, default=None)
    parser.add_argument("--GRF_file", type=str, default="") # file with generative relevant feedback
    parser.add_argument("--with_original", type=bool, default=False)
    parser.add_argument("--rewrite_shuffle", type=bool, default=False)
    parser.add_argument("--rewrite_nums", type=int, default=5)
    parser.add_argument("--GRF_nums", type=int, default=1)

    parser.add_argument("--log_print_steps", type=float, default=1000, help="Percent of steps per epoch to print once.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="num_train_epochs")
    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")

    # yuchen: probably problematic
    parser.add_argument("--num_warmup_steps", type=int, default=0.1, help="Warm up steps.")
    # yuchen: I will use this. 
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warm up ratio w.r.t total training steps.")


    parser.add_argument("--max_query_length", type=int, default=512, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=512, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)

    parser.add_argument("--loss_type", type=str, default="ranking")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train", help="To control how to organize the batch data.")
    #parser.add_argument("--neg_ratio", type=int, help="negative ratio")

    args = parser.parse_args()

    # obsolet for python -m torch.dis...
    # local_rank = args.local_rank

    # pytorch parallel gpu
    if args.n_gpu > 1:
        # initialize the process (with torchrun)
        dist.init_process_group(backend='nccl', init_method='env://')
        # get local rank
        local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = local_rank
        # to get all "cuda" related operations to use the local_rank cuda.
        torch.cuda.set_device(args.local_rank)
        # get local GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
        args.device = device
    else:
        # single GPU
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        args.device = device

    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    #if dist.get_rank() == 0 and args.need_output:
    #    check_dir_exist_or_build([args.output_dir_path], force_emptying=args.force_emptying_dir)
    #    json_dumps_arguments(oj(args.output_dir_path, "parameters.txt"), args)
        

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    train(args)

# python  -m torch.distributed.launch --nproc_per_node 2 xxx.py
