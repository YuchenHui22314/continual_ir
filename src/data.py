import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm, trange
import csv
import random
  
class Topiocqa(Dataset):

    def tokenize(self, text, max_len):
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        return tokens

    def __init__(self, args, tokenizer, filename):

        self.examples = []
        self.tokenizer = tokenizer
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging

        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data, desc="Loading Topiocqa dataset"):
            record = json.loads(line)
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            pos_doc_text = record["pos_docs"][0]  # only use the first positive document
            
            # tokenize hard negative documents
            if 'bm25_hard_neg_docs' in record:
                # bm25_hard_neg_doc = random.choice(record['bm25_hard_neg_docs'])
                # use only one hard negative document
                bm25_hard_neg_doc = record['bm25_hard_neg_docs'][0]
                bm25_hard_neg_doc = self.tokenize(bm25_hard_neg_doc, max_len=args.max_doc_length)
            else:
                bm25_hard_neg_doc = [] 
            
            # tokenize positive documents
            pos_doc_tokens = self.tokenize(pos_doc_text, max_len=args.max_doc_length) 

            # tokenize oracle utterance
            if "oracle_utt_text" in record:
                oracle_utt_text = record["oracle_utt_text"]
                oracle_utt_tokens = self.tokenize(oracle_utt_text, max_len=args.max_query_length)
            else:
                oracle_utt_tokens = []
            

            # We reserve 1 position for [CLS] at the beginning
            max_context_len = args.max_concat_length - 1
            flat_concat = []
            total_len = 0

            # build current utterance tokens
            cur_utt_tokens = self.tokenize(cur_utt_text, max_len=args.max_query_length)

            # goal of this code segement: build the conversation context as:
            # [cls,q1,sep,r1,sep,q2,sep,r2,sep,...,qn,sep,rn,sep]

            # ---- 1. Add current utterance first (most recent) ----
            cur_tokens = cur_utt_tokens[1:]  # remove [CLS], keep [SEP]

            if len(cur_tokens) > max_context_len:
                cur_tokens = cur_tokens[:max_context_len]
                cur_tokens[-1] = self.tokenizer.sep_token_id

            flat_concat.append(cur_tokens)
            total_len += len(cur_tokens)

            # ---- 2. Add historical utterances backward ----
            for j in range(len(ctx_utts_text) - 1, -1, -1):

                # odd index : response, even index : query
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length

                # tokenize with special tokens, then remove [CLS]
                tokens = self.tokenize(ctx_utts_text[j], max_len=max_length)
                tokens = tokens[1:]  # remove [CLS], keep trailing [SEP]

                remaining = max_context_len - total_len
                if remaining <= 0:
                    break

                if len(tokens) <= remaining:
                    flat_concat.append(tokens)
                    total_len += len(tokens)
                else:
                    # need to truncate tokens, but must end with [SEP]
                    truncated = tokens[:remaining]
                    truncated[-1] = self.tokenizer.sep_token_id
                    flat_concat.append(truncated)
                    total_len += len(truncated)
                    break

            # ---- 3. Restore chronological order ----
            flat_concat = flat_concat[::-1]

            # ---- 4. Flatten and prepend [CLS] ----
            flat_concat = [tok for turn in flat_concat for tok in turn]
            flat_tokens = [self.tokenizer.cls_token_id]
            flat_tokens.extend(flat_concat)

            # HARD SAFETY CHECK (debug only, can remove later)
            assert len(flat_tokens) <= args.max_concat_length

            # for j in range(len(ctx_utts_text) - 1, -1, -1):

            #     # odd index : response, even index: query 
            #     if j % 2 == 1:
            #         max_length = args.max_response_length
            #     else:
            #         max_length = args.max_query_length

            #     # with [cls] and [sep]
            #     cls_context = self.tokenize( ctx_utts_text[j], max_len=max_length)
            #     context = cls_context[1:]  # remove [CLS]

            #     remaining = args.max_concat_length - total_length

            #     if len(context) >= remaining:
            #         if remaining > 2:
            #             flat_concat.append(context[:remaining - 2] + [self.tokenizer.sep_token_id])
            #         break
            #     else:
            #         flat_concat.append(context)
            #         total_length += len(context)
            
            # # begin from the earliest turn
            # flat_concat = flat_concat[::-1]  # reverse to make the order correct
            # # [token..., SEP, token..., SEP, ..., token..., SEP]
            # flat_concat = [token_id for turn in flat_concat for token_id in turn]
            # # add [CLS] at the beginning
            # flat_tokens = [self.tokenizer.cls_token_id]
            # flat_tokens.extend(flat_concat)

            self.examples.append(
                {
                    'sample_id': record['sample_id'], 
                    'conversation_tokens': flat_tokens,
                    "neg_doc_tokens": bm25_hard_neg_doc,
                    "pos_doc_tokens": pos_doc_tokens,
                    'oracle_tokens': oracle_utt_tokens,
                }
            )


    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args, pad_token_id=0):

        def pad_and_mask(seqs):
            """
            Pad a list of variable-length sequences to the maximum length
            in the batch and build attention masks.

            Args:
                seqs (List[List[int]])

            Returns:
                padded (Tensor): [B, L]
                mask   (Tensor): [B, L], 1 for real tokens, 0 for padding
            """
            # Convert to tensors
            tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]

            # Pad to batch-longest
            padded = pad_sequence(
                tensors,
                batch_first=True,
                padding_value=pad_token_id
            )

            # Attention mask
            mask = (padded != pad_token_id).long()
            return padded, mask

        def collate_fn(batch):
            """
            Collate function for Topiocqa.

            Responsibilities:
            - Dynamic padding to batch-longest length
            - Build attention masks
            - Convert lists to tensors
            - Keep sample_id as Python objects
            """

            sample_ids = []
            conv_seqs = []
            pos_doc_seqs = []
            neg_doc_seqs = []
            oracle_seqs = []

            # --------
            # Unpack batch
            # --------
            for example in batch:
                sample_ids.append(example["sample_id"])
                conv_seqs.append(example["conversation_tokens"])
                pos_doc_seqs.append(example["pos_doc_tokens"])
                neg_doc_seqs.append(example["neg_doc_tokens"])
                oracle_seqs.append(example["oracle_tokens"])

            # --------
            # Conversation context
            # --------
            conv_padded, conv_mask = pad_and_mask(conv_seqs)

            # --------
            # Positive documents
            # --------
            pos_padded, pos_mask = pad_and_mask(pos_doc_seqs)

            # --------
            # Negative documents (may be empty)
            # --------
            if any(len(s) > 0 for s in neg_doc_seqs):
                neg_padded, neg_mask = pad_and_mask(neg_doc_seqs)
            else:
                B = len(batch)
                neg_padded = torch.zeros((B, 1), dtype=torch.long)
                neg_mask = torch.zeros((B, 1), dtype=torch.long)

            # --------
            # Oracle utterances (may be empty)
            # --------
            if any(len(s) > 0 for s in oracle_seqs):
                oracle_padded, oracle_mask = pad_and_mask(oracle_seqs)
            else:
                B = len(batch)
                oracle_padded = torch.zeros((B, 1), dtype=torch.long)
                oracle_mask = torch.zeros((B, 1), dtype=torch.long)

            return {
                "sample_ids": sample_ids,

                "complex_query": conv_padded,
                "complex_query_mask": conv_mask,

                "pos_docs": pos_padded,
                "pos_docs_mask": pos_mask,

                "neg_docs": neg_padded,
                "neg_docs_mask": neg_mask,

                "oracle_qr": oracle_padded,
                "oracle_qr_mask": oracle_mask,
            }

        return collate_fn
