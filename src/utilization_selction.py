import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

tokenizer = RobertaTokenizer.from_pretrained('../../checkpoints/ad-hoc-ance-msmarco')
model = RobertaModel.from_pretrained('../../checkpoints/ad-hoc-ance-msmarco')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

def compute_batch_fisher(model, queries, documents):
    model.eval()
    
    queries = list(queries)
    documents = list(documents)
    
    similarity = model(queries, documents)  # batch similarity scores
    
    pseudo_labels = torch.ones_like(similarity).to(device)
    
    loss = F.mse_loss(similarity, pseudo_labels, reduction='sum')  
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)

    fisher_scores = []
    for i in range(len(queries)):
        model.zero_grad()
        
        sim_i = model([queries[i]], [documents[i]])
        pseudo_label_i = torch.ones_like(sim_i).to(device)
        loss_i = F.mse_loss(sim_i, pseudo_label_i, reduction='sum')
        
        grads_i = torch.autograd.grad(loss_i, model.parameters(), retain_graph=False)
        fisher_i = sum((g**2).sum() for g in grads_i if g is not None)
        fisher_scores.append(fisher_i.item())

    return fisher_scores

def batch_fisher(batch_query_lists, k=K):
    return [compute_batch_fisher(query_list, k) for query_list in batch_query_lists]

def FIM_data(input_data_path, output_data_path, batch_size):
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
                result = batch_fisher(batch, k=3)
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
FIM_data(input_data_path, output_data_path, batch_size)