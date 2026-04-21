from __future__ import annotations
import os
from pdb import run
import faiss
import pickle
import json
import random
import numpy as np


import csv
import json
import logging

from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
# AdamW was removed from transformers in newer versions; use torch.optim directly
AdamW = torch.optim.AdamW
import torch.nn.functional as F
from IPython import embed

import sys

sys.path += ['../']
# import pandas as pd
# from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
import torch.distributed as dist
from tqdm import tqdm, trange
from os import listdir
from os.path import isfile, join

torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
import re
import shutil
from typing import List, Set, Dict, Tuple, Callable, Iterable, Any

def check_dir_exist_or_build(dir_list, erase_dir_content = None):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
    if erase_dir_content:
        for dir_path in erase_dir_content:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def split_and_padding_neighbor(batch_tensor, batch_len):
    batch_len = batch_len.tolist()
    pad_len = max(batch_len)
    device = batch_tensor.device
    tensor_dim = batch_tensor.size(1)

    batch_tensor = torch.split(batch_tensor, batch_len, dim = 0)
    
    padded_res = []
    for i in range(len(batch_tensor)):
        cur_len = batch_tensor[i].size(0)
        if cur_len < pad_len:
            padded_res.append(torch.cat([batch_tensor[i], 
                                        torch.zeros((pad_len - cur_len, tensor_dim)).to(device)], dim = 0))
        else:
            padded_res.append(batch_tensor[i])

    padded_res = torch.cat(padded_res, dim = 0).view(len(batch_tensor), pad_len, tensor_dim)
    
    return padded_res


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))



def load_collection(collection_file):
    all_passages = ["[INVALID DOC ID]"] * 5000_0000
    ext = collection_file[collection_file.rfind(".") + 1:]
    if ext not in ["jsonl", "tsv"]:
        raise TypeError("Unrecognized file type")
    with open(collection_file, "r") as f:
        if ext == "jsonl":
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                pid = int(obj["id"])
                passage = obj["title"] + "[SEP]" + obj["text"]
                all_passages[pid] = passage
        else:
            for line in f:
                line = line.strip()
                try:
                    line_arr = line.split("\t")
                    pid = int(line_arr[0])
                    passage = line_arr[1].rstrip()
                    all_passages[pid] = passage
                except IndexError:
                    print("bad passage")
                except ValueError:
                    print("bad pid")
    return all_passages


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, fused: bool = False) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if fused:
        # fused AdamW: faster optimizer step on CUDA (PyTorch >= 2.0; A6000 supported)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                                      eps=args.adam_epsilon, fused=True)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def optimizer_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)



class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                # print("yielding record")
                # print(rec)
                yield rec

def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def barrier_array_merge(args,
                        data_array,
                        merge_axis=0,
                        prefix="",
                        load_cache=False,
                        only_load_in_master=False,
                        merge=True):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    if not merge:
        return None

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(args.world_size
                   ):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(
            args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg

def print_res(result_file, gold_file):
    final_scores = {}

    with open(result_file, 'r') as f:
        result_data = json.load(f)
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)

    
    ranks = []
    MRR_score = 0.0
    NDCG_score = 0.0
    #norm = 1 / np.log2(2)
    for i, sample in enumerate(gold_data):
        assert str(sample["conv_id"]) == str(result_data[i]["conv_id"])
        assert str(sample["turn_id"]) == str(result_data[i]["turn_id"])

        gold_ctx = sample["positive_ctxs"][0]
        rank_assigned = False
        for rank, ctx in enumerate(result_data[i]["ctxs"]):
            if ctx["doc_id"] ==  gold_ctx["passage_id"]:
                MRR_score += 1.0 / (rank + 1)
                NDCG_score += 1 / np.log2(rank + 2) #/ max(0.3, norm)
                ranks.append(float(rank + 1))
                rank_assigned = True
                break
        if not rank_assigned:
            ranks.append(1000.0)

    for n in [1, 3, 5, 10, 20, 30 ,50, 100]:
        if len(ranks) == 0:
            score = 0
        else:
            score = len([x for x in ranks if x <= n]) * 100.0 / len(ranks)
        #score = hits_at_n(ranks, n)
        final_scores["R@" + str(n)] = round(score, 2)
    MRR_score = round(MRR_score * 100.0 / len(ranks), 2)
    NDCG_score = round(NDCG_score * 100.0 / len(ranks), 2)
    final_scores["MRR"] = MRR_score
    final_scores["NDCG"] = NDCG_score

    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(json.dumps(final_scores, indent=4))

    return final_scores

STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",  "then", "here", "there", "when", "where", "why", "how", "other", "some", "such", "nor", "not", "only", "own", "same", "now"]
BIG_STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
BIGBIG_STOP_WORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
QUESTION_WORD_LIST = ["what", "when", "why", "who", "how", "where", "whose", "which", "is", "are", "were", "was", "do", "does", "did", "can"]
OTHER_WORD_LIST = ["tell"]


def is_nl_query(query):
    if any([query.lower().startswith(word) for word in QUESTION_WORD_LIST]):
        return True
    return False

def format_nl_query(query):
    query = query.replace("?", "")
    query = query.replace("\\", "")
    query = query.replace("\"", "")
    if is_nl_query(query):
        query = query[0].upper() + query[1:] + "?"
    else:
        query = query[0].upper() + query[1:] + "."
    return query



class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number



#################
### BEIR eval
#################

#from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from models import BeirEvalANCEQueryEncoder

def eval_beir_datasets(
    dataset_list: List[str],
    query_encoder_for_eval: Any,
    query_tokenizer: Any,
    embedding_base_path: str,
    beir_data_path: str,
    batch_size: int,
    device: Any,
    run_id: str, # give beir query embedding file a unique id, so that different models' embeddings won't conflict
    epoch: int
):
    '''
    The code won't work for cqadupstack
    return:
        metric_numbers: dict, key is dataset name, value is ndcg@10 score
    
    '''

    # take query encoder
    query_encoder_for_eval.eval()

    beir_eval_encoder = BeirEvalANCEQueryEncoder(
        query_encoder = query_encoder_for_eval,
        query_encoder_tokenizer = query_tokenizer,
        device = device
    )

    model = DRES(beir_eval_encoder, batch_size=batch_size)
    
    metric_numbers = {}

    for dataset_name in dataset_list:

        embedding_dir = os.path.join(embedding_base_path, dataset_name) 
        split = "test"
        if dataset_name == "msmarco":
            split = "dev"
        corpus, queries, qrels = BeirCustomDataLoader(os.path.join(beir_data_path, dataset_name, dataset_name)).load(split=split) # or split = "train" or "dev"

        retriever = EvaluateRetrieval(model, score_function="dot")

        retrieval_results = retriever.encode_and_retrieve(
            corpus=corpus,
            queries=queries,
            encode_output_path = embedding_dir,
            overwrite=False,  # Set to True if you want to overwrite existing embeddings
            query_filename=f"{run_id}_epoch_{epoch}_queries.pkl"
        )


        ndcg, _map, recall, precision = retriever.evaluate(qrels, retrieval_results, retriever.k_values)

        # take ndcg@10
        metric_numbers[dataset_name] = ndcg["NDCG@10"]
    
    return metric_numbers


def make_fake_corpus(doc_ids):
    """
    Create a fake BEIR-compatible corpus that is safe for
    sorting, length checking, and batching, but contains no text.
    """
    return {
        doc_id: {
            "title": "",
            "text": "",
        }
        for doc_id in doc_ids
    }


class BeirCustomDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load_custom(
        self,
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> tuple[dict[str, str], dict[str, dict[str, int]]]:
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        corpus = make_fake_corpus("hello")
        return corpus, self.queries, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score


#################
### In-Memory FAISS Corpus Loading and Evaluation
### Loads all corpus embeddings once at training startup; per-epoch eval only re-encodes queries.
#################

def load_corpus_into_faiss(embedding_dir: str, embed_dim: int = 768, use_gpu: bool = False):
    """
    Load all corpus embedding blocks from embedding_dir into a single FAISS flat-IP index.
    Supports two file formats:
      BEIR format:     corpus.{i}.pkl  →  tuple (float32 array, list of doc_ids)
      TopiOCQA format: doc_emb_block.{i}.pb + doc_embid_block.{i}.pb

    Args:
        embedding_dir: directory containing the embedding block files
        embed_dim:     embedding dimension (default: 768 for ANCE)
        use_gpu:       if True, shard FAISS index across all available GPUs (requires faiss-gpu)

    Returns:
        (faiss_index, doc_ids):  flat-IP FAISS index + np.array of corresponding doc IDs
    """
    index = faiss.IndexFlatIP(embed_dim)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    all_ids = []

    beir_0     = os.path.join(embedding_dir, "corpus.0.pkl")
    topiocqa_0 = os.path.join(embedding_dir, "doc_emb_block.0.pb")

    if os.path.exists(beir_0):
        # BEIR format: each corpus.{i}.pkl is (embs_array, ids_list)
        block_id = 0
        while True:
            fpath = os.path.join(embedding_dir, f"corpus.{block_id}.pkl")
            if not os.path.exists(fpath):
                break
            with open(fpath, "rb") as f:
                embs, ids = pickle.load(f)
            index.add(np.array(embs, dtype=np.float32))
            all_ids.extend(ids)
            del embs
            block_id += 1
            if block_id % 20 == 0:
                logger.info(f"  load_corpus_into_faiss: {block_id} BEIR blocks "
                            f"({index.ntotal} docs) from {embedding_dir}")

    elif os.path.exists(topiocqa_0):
        # TopiOCQA format: separate emb and id files per block
        block_id = 0
        while True:
            emb_path = os.path.join(embedding_dir, f"doc_emb_block.{block_id}.pb")
            id_path  = os.path.join(embedding_dir, f"doc_embid_block.{block_id}.pb")
            if not os.path.exists(emb_path):
                break
            with open(emb_path, "rb") as f:
                embs = pickle.load(f)
            with open(id_path, "rb") as f:
                ids = pickle.load(f)
            index.add(np.array(embs, dtype=np.float32))
            all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else list(ids))
            del embs
            block_id += 1
            if block_id % 5 == 0:
                logger.info(f"  load_corpus_into_faiss: {block_id} TopiOCQA blocks "
                            f"({index.ntotal} docs) from {embedding_dir}")
    else:
        raise FileNotFoundError(
            f"No recognized embedding blocks in {embedding_dir}. "
            f"Expected corpus.0.pkl (BEIR) or doc_emb_block.0.pb (TopiOCQA)."
        )

    logger.info(f"load_corpus_into_faiss: {index.ntotal} total embeddings "
                f"from {os.path.basename(embedding_dir)}")
    return index, np.array(all_ids)


def _build_topiocqa_query_tokens(
    tokenizer,
    cur_utt_text:        str,
    ctx_utts_text:       list,
    max_query_length:    int = 32,
    max_response_length: int = 32,
    max_concat_length:   int = 512,
) -> list:
    """
    Standalone version of Topiocqa.build_conv_query_tokens from data.py.
    Used in eval_topiocqa() to tokenize valid-set queries without instantiating the dataset class.
    Format: [CLS] cur_q [SEP] ctx_n [SEP] ... ctx_1 [SEP]  (chronological, bounded by max_concat_length)
    """
    def _tok(text, max_len):
        return tokenizer.encode(text, add_special_tokens=True, max_length=max_len, truncation=True)

    max_context_len = max_concat_length - 1  # reserve 1 position for [CLS]
    flat_concat = []
    total_len   = 0

    # Current utterance: tokenize, remove leading [CLS], keep trailing [SEP]
    cur_tokens = _tok(cur_utt_text, max_len=max_query_length)[1:]
    if len(cur_tokens) > max_context_len:
        cur_tokens = cur_tokens[:max_context_len]
        cur_tokens[-1] = tokenizer.sep_token_id
    flat_concat.append(cur_tokens)
    total_len += len(cur_tokens)

    # Historical utterances: add backwards (most recent first), restore order later
    for j in range(len(ctx_utts_text) - 1, -1, -1):
        max_len = max_response_length if j % 2 == 1 else max_query_length
        tokens  = _tok(ctx_utts_text[j], max_len=max_len)[1:]  # remove [CLS]
        remaining = max_context_len - total_len
        if remaining <= 0:
            break
        if len(tokens) <= remaining:
            flat_concat.append(tokens)
            total_len += len(tokens)
        else:
            truncated = tokens[:remaining]
            truncated[-1] = tokenizer.sep_token_id
            flat_concat.append(truncated)
            break

    # Restore chronological order, prepend [CLS]
    flat_concat = flat_concat[::-1]
    flat_tokens = [tokenizer.cls_token_id]
    for seg in flat_concat:
        flat_tokens.extend(seg)
    return flat_tokens


def _compute_full_metrics(qrels: dict, run: dict) -> dict:
    """
    Compute the full set of retrieval metrics for a single dataset using pytrec_eval via BEIR.

    Metrics computed:
        map (MAP@10)
        ndcg_cut.1/3/5/10
        P.1/3/5/10
        recall.5/50/100/1000
        mrr@1/3/5/10

    Args:
        qrels: {qid: {doc_id: relevance}}
        run:   {qid: {doc_id: score}}

    Returns:
        Flat dict of metric_name → float, e.g. {"NDCG@10": 0.42, "MRR@10": 0.38, ...}
    """
    retriever = EvaluateRetrieval(None)
    k_values  = [1, 3, 5, 10, 50, 100, 1000]

    ndcg, _map, recall, precision = retriever.evaluate(qrels, run, k_values)
    mrr = retriever.evaluate_custom(qrels, run, [1, 3, 5, 10], metric="mrr")

    metrics = {}
    # MAP@10
    metrics["MAP@10"] = _map.get("MAP@10", 0.0)
    # NDCG@k
    for k in [1, 3, 5, 10]:
        metrics[f"NDCG@{k}"] = ndcg.get(f"NDCG@{k}", 0.0)
    # Precision@k
    for k in [1, 3, 5, 10]:
        metrics[f"P@{k}"] = precision.get(f"P@{k}", 0.0)
    # Recall@k
    for k in [5, 50, 100, 1000]:
        metrics[f"Recall@{k}"] = recall.get(f"Recall@{k}", 0.0)
    # MRR@k
    for k in [1, 3, 5, 10]:
        metrics[f"MRR@{k}"] = mrr.get(f"MRR@{k}", 0.0)
    return metrics


def eval_topiocqa(
    query_encoder,
    tokenizer,
    test_data_file:      str,
    qrel_file:           str,
    faiss_index,
    doc_ids:             np.ndarray,
    device,
    eval_batch_size:     int  = 64,
    max_query_length:    int  = 32,
    max_response_length: int  = 32,
    max_concat_length:   int  = 512,
    top_k:               int  = 100,
    use_gpu_faiss:       bool = False,
    keep_faiss_on_gpu:   bool = False,
    gpu_index_cache:     dict = None,
    full_eval:           bool = False,
    left_padding:        bool = False,
) -> dict:
    """
    Per-epoch TopiOCQA evaluation using a pre-loaded in-memory FAISS index.
    Only re-encodes queries — no corpus disk I/O.

    Args:
        query_encoder:   current query encoder (temporarily set to eval mode; DDP-unwrapped)
        tokenizer:       query tokenizer
        test_data_file:  topiocqa_valid.jsonl  (fields: Conversation_no, Turn_no, Question, Context)
        qrel_file:       TREC qrel (4 columns, no header): "qid 0 doc_id relevance"
        faiss_index:     pre-loaded FAISS flat-IP CPU index for TopiOCQA corpus
        doc_ids:         np.array of doc ID strings corresponding to FAISS index rows
        device:          torch device for encoding
        top_k:           number of docs to retrieve per query
        use_gpu_faiss:   if True, transfer faiss_index to GPU (sharded) before searching
        keep_faiss_on_gpu: if True AND use_gpu_faiss, cache the GPU index in gpu_index_cache
                           so subsequent epochs reuse it without re-transfer (saves ~21s/epoch)
        gpu_index_cache: mutable dict shared with caller — key "topiocqa" stores GPU index.
                         Pass the same dict every epoch so the cache persists.
        full_eval:       if True, compute the full metric suite (MAP, NDCG@1/3/5/10, P@1/3/5/10,
                         Recall@5/50/100/1000, MRR@1/3/5/10) and return as flat dict.
                         Also forces top_k=1000 to support Recall@1000.
                         Use on the final epoch; log results to wandb summary.

    Returns:
        If full_eval=False: dict {"NDCG@10": float, f"Recall@{top_k}": float, "MRR@10": float}
        If full_eval=True:  flat dict with all metrics (see _compute_full_metrics).
    """
    encoder = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
    encoder.eval()

    # full_eval requires Recall@1000 → force top_k=1000
    if full_eval:
        top_k = 1000

    # Load qrels — TREC 4-column format: qid  0  doc_id  rel  (no header)
    qrels = {}
    with open(qrel_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, doc_id, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[doc_id] = rel

    # Load and tokenize valid queries (same conversation format as training)
    query_ids, query_token_lists = [], []
    with open(test_data_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            qid = f"{record['Conversation_no']}-{record['Turn_no']}"
            if qid not in qrels:
                continue  # skip queries without relevance judgements
            tokens = _build_topiocqa_query_tokens(
                tokenizer,
                cur_utt_text        = record["Question"],
                ctx_utts_text       = record["Context"],
                max_query_length    = max_query_length,
                max_response_length = max_response_length,
                max_concat_length   = max_concat_length,
            )
            query_ids.append(qid)
            query_token_lists.append(tokens)

    # Encode queries in batches (pad within each batch to batch-max length)
    pad_id   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(query_token_lists), eval_batch_size):
            seqs    = query_token_lists[i : i + eval_batch_size]
            max_len = max(len(s) for s in seqs)
            if left_padding:
                # Left-pad: required for FlashAttention 2 (Qwen3). Last real token at position -1.
                rows = [[pad_id] * (max_len - len(s)) + s for s in seqs]
            else:
                rows = [s + [pad_id] * (max_len - len(s)) for s in seqs]
            ids_tensor = torch.tensor(rows, dtype=torch.long).to(device)
            mask = (ids_tensor != pad_id).long()
            embs = encoder(input_ids=ids_tensor, attention_mask=mask)
            embs = F.normalize(embs, p=2, dim=-1)
            all_embs.append(embs.float().cpu().numpy())  # float() handles BF16/FP16 models

    query_embs = np.concatenate(all_embs, axis=0)  # (n_queries, 768)

    # FAISS search — optionally move index to GPU for faster search
    _cache_key = "topiocqa"
    if use_gpu_faiss:
        if keep_faiss_on_gpu and gpu_index_cache is not None and _cache_key in gpu_index_cache:
            # Reuse cached GPU index from a previous epoch
            idx_to_search = gpu_index_cache[_cache_key]
            logger.info("eval_topiocqa: reusing cached GPU FAISS index.")
        else:
            # Transfer CPU index → sharded GPU index
            logger.info("eval_topiocqa: transferring TopiOCQA index to GPU ...")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            idx_to_search = faiss.index_cpu_to_all_gpus(faiss_index, co=co)
            if keep_faiss_on_gpu and gpu_index_cache is not None:
                gpu_index_cache[_cache_key] = idx_to_search
                logger.info("eval_topiocqa: GPU index cached (keep_faiss_on_gpu=True).")
        scores, indices = idx_to_search.search(query_embs.astype(np.float32), top_k)
        if not keep_faiss_on_gpu:
            del idx_to_search  # free GPU memory after search
    else:
        scores, indices = faiss_index.search(query_embs.astype(np.float32), top_k)

    # Build run dict {qid: {doc_id: score}}
    run = {}
    for q_idx, qid in enumerate(query_ids):
        run[qid] = {
            str(doc_ids[int(idx)]): float(scores[q_idx, r])
            for r, idx in enumerate(indices[q_idx])
            if 0 <= int(idx) < len(doc_ids)
        }

    if full_eval:
        # Full metric suite for final-epoch summary logging
        metrics = _compute_full_metrics(qrels, run)
        logger.info(
            f"TopiOCQA full eval: NDCG@10={metrics['NDCG@10']:.4f}  "
            f"Recall@100={metrics['Recall@100']:.4f}  "
            f"MRR@10={metrics['MRR@10']:.4f}  "
            f"MAP@10={metrics['MAP@10']:.4f}"
        )
    else:
        # Lightweight per-epoch eval: NDCG@10, Recall@top_k, MRR@10
        # MAP@10 == MRR@10 for TopiOCQA (exactly one relevant doc per query)
        retriever = EvaluateRetrieval(None)
        ndcg, _map, recall, _ = retriever.evaluate(qrels, run, [10, top_k])
        metrics = {
            "NDCG@10":           ndcg["NDCG@10"],
            f"Recall@{top_k}":   recall[f"Recall@{top_k}"],
            "MRR@10":            _map["MAP@10"],
        }
        logger.info(f"TopiOCQA eval: NDCG@10={metrics['NDCG@10']:.4f}  "
                    f"Recall@{top_k}={metrics[f'Recall@{top_k}']:.4f}  "
                    f"MRR@10={metrics['MRR@10']:.4f}")
    encoder.train()
    return metrics


def build_beir_eval_cache(
    dataset_list:        List[str],
    embedding_base_path: str,
    beir_data_path:      str,
    embed_dim:           int  = 768,
    use_gpu:             bool = False,
) -> dict:
    """
    Load BEIR corpus embeddings into memory once at training startup.
    Returns {dataset_name: (faiss_index, doc_ids, queries_dict, qrels_dict)}.
    Pass the result to eval_beir_from_cache() each epoch.

    Args:
        dataset_list:        e.g. ["climate-fever", "msmarco"]
        embedding_base_path: base dir, each dataset has a subdir with corpus.*.pkl files
        beir_data_path:      base dir for BEIR text data (queries + qrels)
        embed_dim:           embedding dimension (768 for ANCE)
        use_gpu:             if True, shard FAISS indices across GPUs (requires faiss-gpu)
    """
    cache = {}
    for dataset_name in dataset_list:
        logger.info(f"build_beir_eval_cache: loading {dataset_name} corpus into FAISS ...")
        embedding_dir = os.path.join(embedding_base_path, dataset_name)
        faiss_idx, dids = load_corpus_into_faiss(embedding_dir, embed_dim=embed_dim, use_gpu=use_gpu)

        split = "dev" if dataset_name == "msmarco" else "test"
        _, queries, qrels = BeirCustomDataLoader(
            os.path.join(beir_data_path, dataset_name, dataset_name)
        ).load(split=split)

        cache[dataset_name] = (faiss_idx, dids, queries, qrels)
        logger.info(f"build_beir_eval_cache: {dataset_name} ready — "
                    f"{faiss_idx.ntotal} docs, {len(queries)} queries.")
    return cache


def eval_beir_from_cache(
    beir_cache:        dict,
    query_encoder,
    tokenizer,
    device,
    eval_batch_size:   int  = 64,
    max_length:        int  = 64,
    top_k:             int  = 1000,
    use_gpu_faiss:     bool = False,
    keep_faiss_on_gpu: bool = False,
    gpu_index_cache:   dict = None,
    full_eval:         bool = False,
) -> dict:
    """
    Per-epoch BEIR evaluation using pre-loaded in-memory FAISS indices.
    Only re-encodes queries (no corpus disk I/O).

    Args:
        beir_cache:        dict returned by build_beir_eval_cache()
        query_encoder:     current query encoder (temporarily set to eval mode)
        tokenizer:         query tokenizer
        device:            torch device
        eval_batch_size:   batch size for query encoding
        max_length:        max token length for BEIR plain-text queries (default 64)
        top_k:             number of docs to retrieve per query
        use_gpu_faiss:     if True, transfer each corpus's CPU index to GPU before searching
        keep_faiss_on_gpu: if True AND use_gpu_faiss, cache GPU indices in gpu_index_cache
                           so subsequent epochs skip the transfer (~3-20s savings per dataset)
        gpu_index_cache:   mutable dict shared with caller — key = dataset_name → gpu_faiss_index.
                           Pass the same dict every epoch so the cache persists across calls.
        full_eval:         if True, compute the full metric suite per dataset and return nested dict
                           {dataset_name: {metric_name: value}}. Also forces top_k=1000.
                           Use on the final epoch; log results to wandb summary.

    Returns:
        If full_eval=False: {dataset_name: ndcg@10 score}
        If full_eval=True:  {dataset_name: {metric_name: value}} (all metrics)
    """
    encoder = query_encoder.module if hasattr(query_encoder, "module") else query_encoder
    encoder.eval()

    # full_eval requires Recall@1000 → force top_k=1000
    if full_eval:
        top_k = 1000

    results = {}

    for dataset_name, (faiss_idx, doc_ids, queries, qrels) in beir_cache.items():
        query_ids   = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Encode queries (plain text, not conversational)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(query_texts), eval_batch_size):
                batch   = query_texts[i : i + eval_batch_size]
                encoded = tokenizer(batch, max_length=max_length, padding=True,
                                    truncation=True, return_tensors="pt")
                embs = encoder(input_ids=encoded["input_ids"].to(device),
                               attention_mask=encoded["attention_mask"].to(device))
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs.float().cpu().numpy())  # float() handles BF16/FP16 models

        query_embs = np.concatenate(all_embs, axis=0)

        # FAISS search — optionally move index to GPU for faster search
        if use_gpu_faiss:
            if keep_faiss_on_gpu and gpu_index_cache is not None and dataset_name in gpu_index_cache:
                # Reuse GPU index cached from a previous epoch
                idx_to_search = gpu_index_cache[dataset_name]
                logger.info(f"  eval_beir_from_cache {dataset_name}: reusing cached GPU index.")
            else:
                # Transfer CPU index → sharded GPU index
                logger.info(f"  eval_beir_from_cache {dataset_name}: transferring index to GPU ...")
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                idx_to_search = faiss.index_cpu_to_all_gpus(faiss_idx, co=co)
                if keep_faiss_on_gpu and gpu_index_cache is not None:
                    gpu_index_cache[dataset_name] = idx_to_search
                    logger.info(f"  eval_beir_from_cache {dataset_name}: GPU index cached.")
            scores, indices = idx_to_search.search(query_embs.astype(np.float32), top_k)
            if not keep_faiss_on_gpu:
                del idx_to_search  # free GPU memory after search
        else:
            scores, indices = faiss_idx.search(query_embs.astype(np.float32), top_k)

        # Build run dict {qid: {doc_id: score}}
        run = {}
        for q_idx, qid in enumerate(query_ids):
            run[qid] = {
                str(doc_ids[int(idx)]): float(scores[q_idx, r])
                for r, idx in enumerate(indices[q_idx])
                if 0 <= int(idx) < len(doc_ids)
            }

        if full_eval:
            # Full metric suite for final-epoch summary logging
            dataset_metrics = _compute_full_metrics(qrels, run)
            results[dataset_name] = dataset_metrics
            logger.info(
                f"  eval_beir_from_cache {dataset_name} full eval: "
                f"NDCG@10={dataset_metrics['NDCG@10']:.4f}  "
                f"Recall@100={dataset_metrics['Recall@100']:.4f}  "
                f"MRR@10={dataset_metrics['MRR@10']:.4f}  "
                f"MAP@10={dataset_metrics['MAP@10']:.4f}"
            )
        else:
            # Lightweight per-epoch eval: NDCG@10 only
            retriever = EvaluateRetrieval(None)
            ndcg, _, _, _ = retriever.evaluate(qrels, run, [10])
            results[dataset_name] = ndcg["NDCG@10"]
            logger.info(f"  eval_beir_from_cache {dataset_name}: NDCG@10 = {ndcg['NDCG@10']:.4f}")

    encoder.train()
    return results
