import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import KMeans
import json

tokenizer = RobertaTokenizer.from_pretrained('../../checkpoints/ad-hoc-ance-msmarco')
encoder = RobertaModel.from_pretrained('../../checkpoints/ad-hoc-ance-msmarco')
encoder.eval()

K = 3

def encode_queries(queries):
    inputs = tokenizer(queries, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = encoder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

def cluster_queries(query_list, k=K):
    embeddings = encode_queries(query_list)
    if len(query_list) < k:
        result = [[] for _ in range(k)]
        for i, q in enumerate(query_list):
            result[i].append(q)
        return result

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clustered = [[] for _ in range(k)]
    for q, label in zip(query_list, labels):
        clustered[label].append(q)
    return clustered

def batch_cluster(batch_query_lists, k=K):
    return [cluster_queries(query_list, k) for query_list in batch_query_lists]

def cluster_data(input_data_path, output_data_path, batch_size):
    with open(input_data_path, 'r') as f:
        data = f.readlines()
    size = 0
    batch = []
    ids = []
    with open(output_data_path, 'w') as g:
        for line in data:
            record = json.loads(line)
            size += 1
            if "rewrites" in input_data_path:
                text_list = record["rewrites"]
            elif "GRF" in input_data_path:
                text_list = record["GRF"]
            batch.append(text_list)
            ids.append(record["sample_id"])
            #breakpoint()
            # clustering and print
            if size % batch_size == 0:
                result = batch_cluster(batch, k=3)
                print_batch = []
                for i, group in enumerate(result):
                    for index, sublist in enumerate(group):
                        if len(sublist) > 0:
                            print_batch.append(sublist)
                    new_record = {}
                    new_record["sample_id"] = ids[i]
                    if "rewrites" in input_data_path:
                        new_record["cluster_rewrites"] = print_batch
                    elif "GRF" in input_data_path:
                        new_record["cluster_GRF"] = print_batch
                    g.write(json.dumps(new_record))
                    g.write('\n')
                batch = []
                ids = []
            #breakpoint()

                    
batch_size = 1
input_data_path = "datasets/topiocqa/topiocqa_train.json"
output_data_path = "datasets/topiocqa/topiocqa_train_cluster.json"
cluster_data(input_data_path, output_data_path, batch_size)