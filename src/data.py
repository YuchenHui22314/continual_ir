import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def pad_and_mask(seqs, max_length, pad_token_id=0):
    """
    Shared helper to pad a list of variable-length sequences to the maximum length
    in the batch and build attention masks.
    """
    # create a tenror of length max_length, add it to the seqs, do the pad, then remove this extra row
    seqs.append([pad_token_id] * max_length)
    # Convert to tensors

    tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]

    # Pad to batch-longest
    padded = pad_sequence(
        tensors,
        batch_first=True,
        padding_value=pad_token_id
    )

    padded = padded[:-1, :]  # remove the extra row

    # Attention mask
    mask = (padded != pad_token_id).long()
    return padded, mask

class BaseDataset(Dataset):
    """
    Base class containing shared logic for tokenization and collation.
    """
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.examples = []

    def tokenize(self, text, max_len):
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        return tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args, max_length, pad_token_id=0):
        def collate_fn(batch):
            sample_ids = []
            complex_query_seqs = []  # Changed from conv_seqs
            pos_doc_seqs = []
            neg_doc_seqs = []
            oracle_seqs = []

            # Unpack batch
            for example in batch:
                sample_ids.append(example["sample_id"])
                complex_query_seqs.append(example["query_tokens"]) 
                pos_doc_seqs.append(example["pos_doc_tokens"])
                neg_doc_seqs.append(example["neg_doc_tokens"])
                oracle_seqs.append(example["oracle_tokens"])

            # 1. Complex Query (Conversation or Single Query)
            conv_padded, conv_mask = pad_and_mask(complex_query_seqs, max_length, pad_token_id)

            # 2. Positive Documents
            pos_padded, pos_mask = pad_and_mask(pos_doc_seqs, max_length, pad_token_id)

            # 3. Negative Documents
            if any(len(s) > 0 for s in neg_doc_seqs):
                neg_padded, neg_mask = pad_and_mask(neg_doc_seqs, max_length, pad_token_id)
            else:
                B = len(batch)
                neg_padded = torch.zeros((B, 1), dtype=torch.long)
                neg_mask = torch.zeros((B, 1), dtype=torch.long)

            # 4. Oracle Utterances
            if any(len(s) > 0 for s in oracle_seqs):
                oracle_padded, oracle_mask = pad_and_mask(oracle_seqs, max_length, pad_token_id)
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


class Topiocqa(BaseDataset):
    def __init__(self, args, tokenizer, filename):
        super().__init__(args, tokenizer)

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
            pos_doc_text = record["pos_docs"][0] # only use the first positive document
            
            # tokenize hard negative documents
            if 'bm25_hard_neg_docs' in record and len(record['bm25_hard_neg_docs']) > 0:
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
            # goal of this code segement: build the conversation context as:
            # [cls,q1,sep,r1,sep,q2,sep,r2,sep,...,qn,sep,rn,sep]

            # ---- 1. Add current utterance first (most recent) ----
            cur_utt_tokens = self.tokenize(cur_utt_text, max_len=args.max_query_length)
            cur_tokens = cur_utt_tokens[1:] # remove [CLS], keep [SEP]

            if len(cur_tokens) > max_context_len:
                cur_tokens = cur_tokens[:max_context_len]
                cur_tokens[-1] = tokenizer.sep_token_id

            flat_concat.append(cur_tokens)
            total_len += len(cur_tokens)

            # ---- 2. Add historical utterances backward ----
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                # odd index : response, even index : query
                if j % 2 == 1: max_length = args.max_response_length
                else: max_length = args.max_query_length

                # tokenize with special tokens, then remove [CLS]
                tokens = self.tokenize(ctx_utts_text[j], max_len=max_length)
                tokens = tokens[1:] # remove [CLS], keep trailing [SEP]

                remaining = max_context_len - total_len
                if remaining <= 0: break

                if len(tokens) <= remaining:
                    flat_concat.append(tokens)
                    total_len += len(tokens)
                else:
                    # need to truncate tokens, but must end with [SEP]
                    truncated = tokens[:remaining]
                    truncated[-1] = tokenizer.sep_token_id
                    flat_concat.append(truncated)
                    total_len += len(truncated)
                    break

            # ---- 3. Restore chronological order ----
            flat_concat = flat_concat[::-1]

            # ---- 4. Flatten and prepend [CLS] ----
            flat_concat = [tok for turn in flat_concat for tok in turn]
            flat_tokens = [tokenizer.cls_token_id]
            flat_tokens.extend(flat_concat)

            # HARD SAFETY CHECK (debug only, can remove later)
            assert len(flat_tokens) <= args.max_concat_length

            self.examples.append({
                'sample_id': record['sample_id'], 
                'query_tokens': flat_tokens, # Unified key
                "neg_doc_tokens": bm25_hard_neg_doc,
                "pos_doc_tokens": pos_doc_tokens,
                'oracle_tokens': oracle_utt_tokens,
            })


class MSMARCODataset(BaseDataset):
    def __init__(self, args, tokenizer, filename):
        super().__init__(args, tokenizer)
        
        with open(filename, encoding="utf-8") as f:
            # Assuming jsonl format, read line by line
            data = f.readlines()
            
        n = len(data)
        # for easier debugging with small data percent
        if args.use_data_percent < 1.0:
            args.use_data_percent *= 0.1
        n = int(args.use_data_percent * n) 
        
        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        for line in tqdm(data, desc="Loading MSMARCO dataset"):
            record = json.loads(line)
            
            # 1. Query Processing
            # MSMARCO does not have conversation history, so 'cur_utt_text' is just the query.
            query_text = record['query']
            # Direct tokenization, includes [CLS] and [SEP]
            query_tokens = self.tokenize(query_text, max_len=args.max_query_length)

            # 2. Positive Documents Processing
            pos_docs_list = record.get("pos_docs", [])
            if len(pos_docs_list) > 0:
                pos_doc_text = pos_docs_list[0]
                pos_doc_tokens = self.tokenize(pos_doc_text, max_len=args.max_doc_length)
            else:
                # Handle edge case: no positive document (should not happen in training data)
                pos_doc_tokens = []

            # 3. Hard Negative Documents Processing
            neg_docs_list = record.get("bing_hard_neg_docs", [])
            if len(neg_docs_list) > 0:
                hard_neg_doc_text = neg_docs_list[0]
                hard_neg_doc_tokens = self.tokenize(hard_neg_doc_text, max_len=args.max_doc_length)
            else:
                hard_neg_doc_tokens = []

            # 4. Oracle Processing
            # MSMARCO does not have oracle query rewrites, so we set this to empty.
            oracle_utt_tokens = []

            self.examples.append({
                'sample_id': record['query_id'], 
                'query_tokens': query_tokens, # Unified key matches Topiocqa
                "neg_doc_tokens": hard_neg_doc_tokens,
                "pos_doc_tokens": pos_doc_tokens,
                'oracle_tokens': oracle_utt_tokens,
            })
# import json
# import random

# import torch
# from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm
  
# class Topiocqa(Dataset):

#     def tokenize(self, text, max_len):
#         tokens = self.tokenizer.encode(
#             text,
#             add_special_tokens=True,
#             max_length=max_len,
#             truncation=True
#         )
#         return tokens

#     def __init__(self, args, tokenizer, filename):

#         self.examples = []
#         self.tokenizer = tokenizer
#         with open(filename, encoding="utf-8") as f:
#             data = f.readlines()
#         n = len(data)
#         n = int(args.use_data_percent * n)  
#         # randomly sample n samples for deugging

#         if n < len(data):
#            random.seed(args.seed)
#            data = random.sample(data, n)

#         for line in tqdm(data, desc="Loading Topiocqa dataset"):
#             record = json.loads(line)
#             cur_utt_text = record['cur_utt_text']
#             ctx_utts_text = record['ctx_utts_text']
#             pos_doc_text = record["pos_docs"][0]  # only use the first positive document
            
#             # tokenize hard negative documents
#             if 'bm25_hard_neg_docs' in record:
#                 # bm25_hard_neg_doc = random.choice(record['bm25_hard_neg_docs'])
#                 # use only one hard negative document
#                 bm25_hard_neg_doc = record['bm25_hard_neg_docs'][0]
#                 bm25_hard_neg_doc = self.tokenize(bm25_hard_neg_doc, max_len=args.max_doc_length)
#             else:
#                 bm25_hard_neg_doc = [] 
            
#             # tokenize positive documents
#             pos_doc_tokens = self.tokenize(pos_doc_text, max_len=args.max_doc_length) 

#             # tokenize oracle utterance
#             if "oracle_utt_text" in record:
#                 oracle_utt_text = record["oracle_utt_text"]
#                 oracle_utt_tokens = self.tokenize(oracle_utt_text, max_len=args.max_query_length)
#             else:
#                 oracle_utt_tokens = []
            

#             # We reserve 1 position for [CLS] at the beginning
#             max_context_len = args.max_concat_length - 1
#             flat_concat = []
#             total_len = 0

#             # build current utterance tokens
#             cur_utt_tokens = self.tokenize(cur_utt_text, max_len=args.max_query_length)

#             # goal of this code segement: build the conversation context as:
#             # [cls,q1,sep,r1,sep,q2,sep,r2,sep,...,qn,sep,rn,sep]

#             # ---- 1. Add current utterance first (most recent) ----
#             cur_tokens = cur_utt_tokens[1:]  # remove [CLS], keep [SEP]

#             if len(cur_tokens) > max_context_len:
#                 cur_tokens = cur_tokens[:max_context_len]
#                 cur_tokens[-1] = self.tokenizer.sep_token_id

#             flat_concat.append(cur_tokens)
#             total_len += len(cur_tokens)

#             # ---- 2. Add historical utterances backward ----
#             for j in range(len(ctx_utts_text) - 1, -1, -1):

#                 # odd index : response, even index : query
#                 if j % 2 == 1:
#                     max_length = args.max_response_length
#                 else:
#                     max_length = args.max_query_length

#                 # tokenize with special tokens, then remove [CLS]
#                 tokens = self.tokenize(ctx_utts_text[j], max_len=max_length)
#                 tokens = tokens[1:]  # remove [CLS], keep trailing [SEP]

#                 remaining = max_context_len - total_len
#                 if remaining <= 0:
#                     break

#                 if len(tokens) <= remaining:
#                     flat_concat.append(tokens)
#                     total_len += len(tokens)
#                 else:
#                     # need to truncate tokens, but must end with [SEP]
#                     truncated = tokens[:remaining]
#                     truncated[-1] = self.tokenizer.sep_token_id
#                     flat_concat.append(truncated)
#                     total_len += len(truncated)
#                     break

#             # ---- 3. Restore chronological order ----
#             flat_concat = flat_concat[::-1]

#             # ---- 4. Flatten and prepend [CLS] ----
#             flat_concat = [tok for turn in flat_concat for tok in turn]
#             flat_tokens = [self.tokenizer.cls_token_id]
#             flat_tokens.extend(flat_concat)

#             # HARD SAFETY CHECK (debug only, can remove later)
#             assert len(flat_tokens) <= args.max_concat_length


#             self.examples.append(
#                 {
#                     'sample_id': record['sample_id'], 
#                     'conversation_tokens': flat_tokens,
#                     "neg_doc_tokens": bm25_hard_neg_doc,
#                     "pos_doc_tokens": pos_doc_tokens,
#                     'oracle_tokens': oracle_utt_tokens,
#                 }
#             )


    
#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):
#         return self.examples[item]

#     @staticmethod
#     def get_collate_fn(args, pad_token_id=0):

#         def pad_and_mask(seqs):
#             """
#             Pad a list of variable-length sequences to the maximum length
#             in the batch and build attention masks.

#             Args:
#                 seqs (List[List[int]])

#             Returns:
#                 padded (Tensor): [B, L]
#                 mask   (Tensor): [B, L], 1 for real tokens, 0 for padding
#             """
#             # Convert to tensors
#             tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]

#             # Pad to batch-longest
#             padded = pad_sequence(
#                 tensors,
#                 batch_first=True,
#                 padding_value=pad_token_id
#             )

#             # Attention mask
#             mask = (padded != pad_token_id).long()
#             return padded, mask

#         def collate_fn(batch):
#             """
#             Collate function for Topiocqa.

#             Responsibilities:
#             - Dynamic padding to batch-longest length
#             - Build attention masks
#             - Convert lists to tensors
#             - Keep sample_id as Python objects
#             """

#             sample_ids = []
#             conv_seqs = []
#             pos_doc_seqs = []
#             neg_doc_seqs = []
#             oracle_seqs = []

#             # --------
#             # Unpack batch
#             # --------
#             for example in batch:
#                 sample_ids.append(example["sample_id"])
#                 conv_seqs.append(example["conversation_tokens"])
#                 pos_doc_seqs.append(example["pos_doc_tokens"])
#                 neg_doc_seqs.append(example["neg_doc_tokens"])
#                 oracle_seqs.append(example["oracle_tokens"])

#             # --------
#             # Conversation context
#             # --------
#             conv_padded, conv_mask = pad_and_mask(conv_seqs)

#             # --------
#             # Positive documents
#             # --------
#             pos_padded, pos_mask = pad_and_mask(pos_doc_seqs)

#             # --------
#             # Negative documents (may be empty)
#             # --------
#             if any(len(s) > 0 for s in neg_doc_seqs):
#                 neg_padded, neg_mask = pad_and_mask(neg_doc_seqs)
#             else:
#                 B = len(batch)
#                 neg_padded = torch.zeros((B, 1), dtype=torch.long)
#                 neg_mask = torch.zeros((B, 1), dtype=torch.long)

#             # --------
#             # Oracle utterances (may be empty)
#             # --------
#             if any(len(s) > 0 for s in oracle_seqs):
#                 oracle_padded, oracle_mask = pad_and_mask(oracle_seqs)
#             else:
#                 B = len(batch)
#                 oracle_padded = torch.zeros((B, 1), dtype=torch.long)
#                 oracle_mask = torch.zeros((B, 1), dtype=torch.long)

#             return {
#                 "sample_ids": sample_ids,

#                 "complex_query": conv_padded,
#                 "complex_query_mask": conv_mask,

#                 "pos_docs": pos_padded,
#                 "pos_docs_mask": pos_mask,

#                 "neg_docs": neg_padded,
#                 "neg_docs_mask": neg_mask,

#                 "oracle_qr": oracle_padded,
#                 "oracle_qr_mask": oracle_mask,
#             }

#         return collate_fn



# class MSMARCODataset(Dataset):

#     def tokenize(self, text, max_len):
#         tokens = self.tokenizer.encode(
#             text,
#             add_special_tokens=True, # Automatically add [CLS] ... [SEP]
#             max_length=max_len,
#             truncation=True
#         )
#         return tokens

#     def __init__(self, args, tokenizer, filename):

#         self.examples = []
#         self.tokenizer = tokenizer
        
#         with open(filename, encoding="utf-8") as f:
#             # Assuming jsonl format, read line by line
#             data = f.readlines()
            
#         n = len(data)
#         n = int(args.use_data_percent * n)  
        
#         # Randomly sample n samples for debugging or fractional usage
#         if n < len(data):
#             random.seed(args.seed)
#             data = random.sample(data, n)

#         for line in tqdm(data, desc="Loading MSMARCO dataset"):
#             record = json.loads(line)
            
#             # 1. Query Processing
#             # MSMARCO does not have conversation history, so 'cur_utt_text' is just the query.
#             query_text = record['query']
            
#             # Direct tokenization, includes [CLS] and [SEP]
#             # In Topiocqa this was 'conversation_tokens', here it is just 'query_tokens'
#             query_tokens = self.tokenize(query_text, max_len=args.max_query_length)

#             # 2. Positive Documents Processing
#             # According to the preprocessing script, pos_docs is a list. We take the first one.
#             pos_docs_list = record.get("pos_docs", [])
#             if len(pos_docs_list) > 0:
#                 pos_doc_text = pos_docs_list[0]
#                 pos_doc_tokens = self.tokenize(pos_doc_text, max_len=args.max_doc_length)
#             else:
#                 # Handle edge case: no positive document (should not happen in training data)
#                 pos_doc_tokens = []

#             # 3. Hard Negative Documents Processing
#             # Corresponds to 'bing_hard_neg_docs' from the preprocessing script
#             neg_docs_list = record.get("bing_hard_neg_docs", [])
#             if len(neg_docs_list) > 0:
#                 # Strategy: take the first one. Alternatively: random.choice(neg_docs_list)
#                 hard_neg_doc_text = neg_docs_list[0]
#                 hard_neg_doc_tokens = self.tokenize(hard_neg_doc_text, max_len=args.max_doc_length)
#             else:
#                 hard_neg_doc_tokens = []

#             # 4. Oracle Processing
#             # MSMARCO does not have oracle query rewrites, so we set this to empty.
#             # The collate_fn will handle this by padding with 0s.
#             oracle_utt_tokens = []

#             self.examples.append(
#                 {
#                     'sample_id': record['query_id'],  # Maps to MSMARCO query_id
#                     'query_tokens': query_tokens,     # Renamed from 'conversation_tokens'
#                     "neg_doc_tokens": hard_neg_doc_tokens,
#                     "pos_doc_tokens": pos_doc_tokens,
#                     'oracle_tokens': oracle_utt_tokens,
#                 }
#             )

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):
#         return self.examples[item]

#     @staticmethod
#     def get_collate_fn(args, pad_token_id=0):
        
#         def pad_and_mask(seqs):
#             """
#             Pad a list of variable-length sequences to the maximum length
#             in the batch and build attention masks.
#             """
#             # Convert to tensors
#             tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]

#             # Pad to batch-longest
#             padded = pad_sequence(
#                 tensors,
#                 batch_first=True,
#                 padding_value=pad_token_id
#             )

#             # Attention mask
#             mask = (padded != pad_token_id).long()
#             return padded, mask

#         def collate_fn(batch):
#             sample_ids = []
#             query_seqs = []
#             pos_doc_seqs = []
#             neg_doc_seqs = []
#             oracle_seqs = []

#             # Unpack batch
#             for example in batch:
#                 sample_ids.append(example["sample_id"])
#                 query_seqs.append(example["query_tokens"]) # Changed key reference
#                 pos_doc_seqs.append(example["pos_doc_tokens"])
#                 neg_doc_seqs.append(example["neg_doc_tokens"])
#                 oracle_seqs.append(example["oracle_tokens"])

#             # 1. Complex Query (Mapping query_seqs to complex_query for compatibility)
#             query_padded, query_mask = pad_and_mask(query_seqs)

#             # 2. Positive docs
#             pos_padded, pos_mask = pad_and_mask(pos_doc_seqs)

#             # 3. Negative docs (Handle potential empty lists)
#             if any(len(s) > 0 for s in neg_doc_seqs):
#                 neg_padded, neg_mask = pad_and_mask(neg_doc_seqs)
#             else:
#                 B = len(batch)
#                 neg_padded = torch.zeros((B, 1), dtype=torch.long)
#                 neg_mask = torch.zeros((B, 1), dtype=torch.long)

#             # 4. Oracle (Always empty for MSMARCO, handled here)
#             if any(len(s) > 0 for s in oracle_seqs):
#                 oracle_padded, oracle_mask = pad_and_mask(oracle_seqs)
#             else:
#                 B = len(batch)
#                 oracle_padded = torch.zeros((B, 1), dtype=torch.long)
#                 oracle_mask = torch.zeros((B, 1), dtype=torch.long)

#             # Return dictionary with keys matching Topiocqa for model compatibility
#             return {
#                 "sample_ids": sample_ids,

#                 "complex_query": query_padded,
#                 "complex_query_mask": query_mask,

#                 "pos_docs": pos_padded,
#                 "pos_docs_mask": pos_mask,

#                 "neg_docs": neg_padded,
#                 "neg_docs_mask": neg_mask,

#                 "oracle_qr": oracle_padded,
#                 "oracle_qr_mask": oracle_mask,
#             }

#         return collate_fn