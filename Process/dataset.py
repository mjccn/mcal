import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import json
from torch_geometric.utils import degree
from torch_scatter import scatter


# global
label2id = {
    "unverified": 0,
    "non-rumor": 1,
    "true": 2,
    "false": 3,
}


def random_pick(list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

def degree_drop_weights(edge_index):
    edge_index = np.array(edge_index)
    row = torch.from_numpy(edge_index[0])
    row = row.long()
    deg = degree(row)
    deg_col = deg[edge_index[0]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    return weights


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(torch.from_numpy(edge_index[0]))
    x = torch.ones((num_nodes,)).to(torch.from_numpy(edge_index)).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, torch.from_numpy(edge_index[1]), reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    edge_index = np.array(edge_index)
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    edge_index = np.array(edge_index)
    drop_edge_index = edge_index[:, sel_mask]
    return drop_edge_index


class RumorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class GraphDataset(Dataset):
    def __init__(self, fold_x, droprate):

        self.fold_x = fold_x
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]

        # ====================================edgeindex========================================
        with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)
        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index

        with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf

        init_row = list(edgeindex[0])
        init_col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row = init_row + burow
        col = init_col + bucol

        init_edgeindex = [row, col]

        # ==================================- dropping + adding + misplacing -===================================#

        choose_list = [1, 2]  # 1-drop 2-add 3-misplace
        probabilities = [0.5, 0.5]  # T15: probabilities = [0.5,0.3,0.2]
        choose_num1 = random_pick(choose_list, probabilities)
        choose_num2 = random_pick(choose_list, probabilities)
        if self.droprate > 0:
            if choose_num1 == 1:

                
                weights = degree_drop_weights(init_edgeindex) 
                edgeindex_pos1 = drop_edge_weighted(init_edgeindex, weights, 0.4, threshold=0.7)

                # length = len(row)
                # poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                # poslist = sorted(poslist)
                # row2 = list(np.array(row)[poslist])
                # col2 = list(np.array(col)[poslist])
                # edgeindex_pos1 = [row2, col2]


            elif choose_num1 == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                edgeindex_pos1 = [row2, col2]

            if choose_num2 == 1:

                weights = degree_drop_weights(init_edgeindex) 
                edgeindex_pos2 = drop_edge_weighted(init_edgeindex, weights, 0.4, threshold=0.7)

                # length = len(row)
                # poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                # poslist = sorted(poslist)
                # row2 = list(np.array(row)[poslist])
                # col2 = list(np.array(col)[poslist])
                # edgeindex_pos2 = [row2, col2]



            elif choose_num2 == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                edgeindex_pos2 = [row2, col2]


        else:
            # new_edgeindex = [row, col]
            edgeindex_pos1 = [row, col]
            edgeindex_pos2 = [row, col]

        # =========================================X===============================================
        with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f0:
            json_inf0 = json.load(j_f0)

        x0 = json_inf0[id]
        x0 = np.array(x0)

        with open('./bert_w2c/T15/t15_mask_015/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)

        x_list = json_inf[id]
        x = np.array(x_list)

        with open('./data/label_15.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        # y = np.array(y)
        if self.droprate > 0:
            if choose_num1 == 1:
                zero_list = [0] * 768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list

                x2 = np.array(x_list)
                x0 = x2
            elif choose_num2 == 1:
                zero_list = [0] * 768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list

                x2 = np.array(x_list)
                x = x2

        return Data(
            x_init=torch.tensor(x0, dtype=torch.float32),
            x0=torch.tensor(x0, dtype=torch.float32),
            x=torch.tensor(x, dtype=torch.float32),
            init_edgeindex=torch.LongTensor(init_edgeindex),
            edge_index1=torch.LongTensor(edgeindex_pos1),
            edge_index2=torch.LongTensor(edgeindex_pos2),
            y1=torch.LongTensor([y]),
            y2=torch.LongTensor([y]))


class test_GraphDataset(Dataset):
    def __init__(self, fold_x, droprate):

        self.fold_x = fold_x
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        # ====================================edgeindex==============================================
        with open('./data/twitter15/' + id + '/after_tweets.pkl', 'rb') as t:
            tweets = pickle.load(t)

        dict = {}
        for index, tweet in enumerate(tweets):
            dict[tweet] = index

        with open('./data/twitter15/' + id + '/after_structure.pkl', 'rb') as f:
            inf = pickle.load(f)

        inf = inf[1:]
        new_inf = []
        for pair in inf:
            new_pair = []
            for E in pair:
                if E == 'ROOT':
                    break
                E = dict[E]
                new_pair.append(E)
            if E != 'ROOT':
                new_inf.append(new_pair)
        new_inf = np.array(new_inf).T
        edgeindex = new_inf

        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)

        new_edgeindex2 = [row, col]

        # =========================================X====================================================
        with open('./bert_w2c/T15/t15_mask_00/' + id + '.json', 'r') as j_f:
            json_inf = json.load(j_f)

        x = json_inf[id]
        x = np.array(x)

        with open('./data/label_15.json', 'r') as j_tags:
            tags = json.load(j_tags)

        y = label2id[tags[id]]
        # y = np.array(y)

        return Data(
            x_init=torch.tensor(x, dtype=torch.float32),
            x0=torch.tensor(x, dtype=torch.float32),
            x=torch.tensor(x, dtype=torch.float32),
            init_edgeindex=torch.LongTensor(new_edgeindex2),
            edge_index1=torch.LongTensor(new_edgeindex2),
            edge_index2=torch.LongTensor(new_edgeindex2),
            y1=torch.LongTensor([y]),
            y2=torch.LongTensor([y]))
