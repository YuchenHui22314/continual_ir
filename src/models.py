import sys
from typing import Dict, List, Any, Optional

from tqdm import tqdm

sys.path += ['../']
import torch
from torch import nn
import numpy as np
import math
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          T5Tokenizer, T5Model, T5ForConditionalGeneration, T5Config, T5EncoderModel,
                          GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel,
                          BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, BertForTokenClassification,
                          DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
                          DPRContextEncoder, DPRQuestionEncoder)

import torch.nn.functional as F
import time


# ANCE model

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



class BeirEvalANCEQueryEncoder:
    def __init__(
        self,
        query_encoder: Any,
        query_encoder_tokenizer: Any,
        device: Optional[torch.device],
        max_length_query: int = 512,
        max_length_doc: int = 512,
    ):
        """
        Asymmetric ANCE encoder:
        - one encoder for queries

        Args:
            query_encoder (Any): Query encoder model
            query_encoder_tokenizer (Any): Query encoder tokenizer
            max_length_query (int): max length for query encoding
            max_length_doc (int): max length for document encoding
        """
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

        # -------------------------
        # Query encoder
        # -------------------------
        self.query_tokenizer = query_encoder_tokenizer
        self.query_encoder = query_encoder
        self.query_encoder.eval()


    def encode_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode queries using the query encoder.
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(queries), batch_size),
                desc="Encoding Queries (Asymmetric)",
            ):
                batch = queries[i : i + batch_size]
                encoded = self.query_tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                dense = self.query_encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                dense = F.normalize(dense, p=2, dim=-1)
                embeddings.append(dense.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode documents using the passage encoder.
        """
        # BEIR-style preprocessing: title + text
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc="Encoding Corpus (Asymmetric)",
            ):
                batch = texts[i : i + batch_size]
                encoded = self.passage_tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                dense = self.passage_encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                dense = F.normalize(dense, p=2, dim=-1)
                embeddings.append(dense.cpu().numpy())

        return np.concatenate(embeddings, axis=0)