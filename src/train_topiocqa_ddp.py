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

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def save_model(args, model_output_path, model, query_tokenizer, optimizer, scheduler, cur_step, cur_epoch, best_loss):
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
        "epoch": cur_epoch,
        "best_loss": best_loss,
    }, oj(output_dir, "trainer_state.pt"))

    logger.info(f"Saved checkpoint at {output_dir}")


def cal_kd_loss(query_embs, oracle_query_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, oracle_query_embs)

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs=None):
    # get batch size
    batch_size = query_embs.size(0)

    # relevance scores: B * B
    pos_scores = query_embs @ pos_doc_embs.T  # [B, B]

    if neg_doc_embs is not None:
        # B * 1 hard negative scores
        neg_scores = torch.sum(query_embs * neg_doc_embs, dim=1, keepdim=True) 
        # B * (B + 1), scores = in_batch negatives + 1 BM25 hard negative
        score_mat = torch.cat([pos_scores, neg_scores], dim=1)
    else:
        score_mat = pos_scores

    # where exist the positive documents (the diagonal elements)
    labels = torch.arange(batch_size, device=query_embs.device)

    return nn.CrossEntropyLoss()(score_mat, labels)


def train(args):
    # -1 
    is_main_process = (args.n_gpu == 1) or (dist.get_rank() == 0)

    # 0. load pre-encoded pos, neg, oracle embeddings if any
    if args.pos_neg_embedding_file is not None:
        # dict: sample_id -> (pos_emb, neg_emb, oracle_emb)
        sample_emb_table = torch.load(
            args.pos_neg_embedding_file,
            map_location= "cpu"
            ) 

        '''
        sample_id_1": {
            "pos":    Tensor[dim],
            "neg":    Tensor[dim] | None,
            "oracle": Tensor[dim] | None
        },
    '''

    # 1. Load query and doc encoders
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    query_tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    query_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    # if we did not encode the pos neg and oracle embeddings beforehand
    if args.pos_neg_embedding_file is None:
        passage_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    query_encoder = DDP(query_encoder, device_ids = [args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.n_gpu > 1:
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
        drop_last = True,
        pin_memory=True
        )

    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))

    # 3. optimizer
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)

    # 4. warmup: yuchen: warm up steps should usually be set to 0.06* total steps. 
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
    args.log_print_steps = max(1, int(args.log_print_ratio * num_steps_per_epoch))

    # if we want to resume
    if args.resume_from_checkpoint is not None:
        ckpt = args.resume_from_checkpoint

        # 1) Load model weights
        model_to_load = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
        model_to_load.load_state_dict(
            torch.load(os.path.join(ckpt, "pytorch_model.bin"),map_location="cpu")
        )

        # 2) Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(ckpt, "optimizer.pt"),map_location="cpu"))
        scheduler.load_state_dict(torch.load(os.path.join(ckpt, "scheduler.pt"),map_location="cpu"))

        # 3) Load trainer state (global step, best loss)
        state = torch.load(os.path.join(ckpt, "trainer_state.pt"), map_location="cpu")
        cur_step = state["step"]
        best_loss = state["best_loss"]
        epoch = state["epoch"]
        start_epoch = epoch + 1  # next epoch to run

        logger.info(f"Resumed from checkpoint {ckpt}: step={cur_step}, best_loss={best_loss}")

    else:
        cur_step = 0
        start_epoch = 0
        best_loss = float("inf")


    ######################################
    #### Main Training Loop: For each Epoch
    ######################################
    epoch_iterator = trange(
        start_epoch,
        args.num_train_epochs,
        desc="Epoch"
    )

    for epoch in epoch_iterator:

        # yuchen: only train query encoder, freeze doc encoder
        query_encoder.train()

        # yuchen: necessary to make the distribution of samples correct
        if args.n_gpu > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch in tqdm(train_loader,  desc="Step"):
            
            #########################
            ### Take inputs
            #########################
            sample_ids = batch['sample_ids']  # B

            complex_query = batch['complex_query'].to(args.device, non_blocking=True) 
            complex_query_mask = batch['complex_query_mask'].to(args.device, non_blocking=True)

            # yuchen: or optimizer.zero_grad()
            # query_encoder.zero_grad()
            optimizer.zero_grad()

            complex_query_embs = query_encoder(complex_query, complex_query_mask)  # B * dim


            ###############################
            ### calcualte (get) embeddings. 
            ###############################

            # if not pre-encoded, encode on the fly
            if args.pos_neg_embedding_file is None:

                pos_docs = batch['pos_docs'].to(args.device, non_blocking=True)
                pos_docs_mask = batch['pos_docs_mask'].to(args.device, non_blocking=True)

                # yuchen: in DPR paper, it is shown that only one BM25 hard negative can be helpful, while two do not help further.
                neg_docs = batch['neg_docs'].to(args.device, non_blocking=True)
                neg_docs_mask = batch['neg_docs_mask'].to(args.device, non_blocking=True)
            
                if "kd" in args.loss_type: 
                    oracle_query = batch['oracle_qr'].to(args.device, non_blocking=True)
                    oracle_query_mask = batch['oracle_qr_mask'].to(args.device, non_blocking=True)
                passage_encoder.eval()
                with torch.no_grad():
                # doc encoder's parameters are frozen
                    pos_doc_embs = passage_encoder(pos_docs, pos_docs_mask).detach()  # B * dim
                    neg_doc_embs = passage_encoder(neg_docs, neg_docs_mask).detach()  # (B * neg_ratio) * dim,      

                    if "kd" in args.loss_type:
                        oracle_utt_embs = passage_encoder(oracle_query, oracle_query_mask).detach()

            else: # if we have pre-encoded pos, neg, oracle embeddings

                pos_doc_embs = torch.stack(
                    [sample_emb_table[sid]["pos"] for sid in sample_ids],
                    dim=0
                ).to(args.device)   # [B, dim]

                if sample_emb_table[sample_ids[0]]["neg"] is not None:
                    neg_doc_embs = torch.stack(
                        [sample_emb_table[sid]["neg"] for sid in sample_ids],
                        dim=0
                    ).to(args.device)
                else:
                    neg_doc_embs = None


                if "kd" in args.loss_type:
                    oracle_utt_embs = torch.stack(
                        [sample_emb_table[sid]["oracle"] for sid in sample_ids],
                        dim=0
                    ).to(args.device)
            
            # end if: pre-encoded pos neg oracle embeddings


            ranking_loss = cal_ranking_loss(complex_query_embs, pos_doc_embs, neg_doc_embs)
            if "kd" in args.loss_type:
                kd_loss = cal_kd_loss(complex_query_embs, oracle_utt_embs)
                loss = ranking_loss + kd_loss
            else:
                loss = ranking_loss

            loss.backward()
            # grad norm should be done after loss.backward and before optimizer.step
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            # after each step(batch), update the learning rate
            scheduler.step()
            
            loss = loss.item()

            # print info
            if is_main_process and cur_step % args.log_print_steps == 0:
                logger.info("Epoch = {}, Current Step = {}, Total Step = {}, Loss = {}".format(
                    epoch,
                    cur_step,
                    total_training_steps,
                    round(loss, 7))
                    )
            #if dist.get_rank() == 0:
            #    log_writer.add_scalar("train_{}_loss".format(args.loss_type), loss, cur_step)
            cur_step += 1    # avoid saving the model of the first step.
            # Save model
            if is_main_process and best_loss > loss:
                save_model(
                    args,
                    args.model_output_path,
                    query_encoder,
                    query_tokenizer,
                    optimizer,
                    scheduler,
                    cur_step,
                    epoch,
                    loss
                )
                best_loss = loss
                

        # for each "epoch", wait for all processes to finish
        if args.n_gpu > 1:
            dist.barrier()
    logger.info("Training finish!")          
    #if dist.get_rank() == 0:   
    #    log_writer.close()
       

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local-rank', type=int, default=-1, metavar='N', help='Local process rank.')  # is useful if using python -m torch.distributed.luanch xxxxx 

    parser.add_argument('--n_gpu', type=int, default=4, help='The number of used GPU.')
    parser.add_argument("--pretrained_encoder_path", type=str, default="/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234")
    parser.add_argument( "--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory to resume training from")
    parser.add_argument("--training_data_file", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/topiocqa/topiocqa_train_oracle.jsonl")
    parser.add_argument("--pos_neg_embedding_file", type=str, default=None)#"/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/topiocqa_pos_neg_docs_ance/embeddings.npy")
    parser.add_argument("--model_output_path", type=str, default="/data/rech/huiyuche/huggingface/continual_ir/topiocqa")

    parser.add_argument("--loss_type", type=str, default="ranking", help="The type of loss to use. Options: kd, kd+ranking, ranking")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="num_train_epochs")
    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="weight_decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")

    # yuchen: probably problematic
    parser.add_argument("--num_warmup_steps", type=int, default=0.06, help="Warm up steps.")
    # yuchen: I will use this. 
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warm up ratio w.r.t total training steps.")


    # complementary settings
    parser.add_argument("--log_print_ratio", type=float, default=0.1, help="Percent of steps per epoch to print once.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=512, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)


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
