import random
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils.convert import to_networkx
from sklearn.model_selection import StratifiedKFold
import pdb
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def k_fold(dataset, folds, epoch_select):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))
    
    return train_indices, test_indices, val_indices


def process_labels(data_y):
    # new_data_list = []
    labels = data_y[:, :3]
    if torch.isnan(labels).any():
        data_y = torch.full_like(data_y[:, :1], float('nan'))
        # print('data.y', data.y)
    else:
        mask = ~torch.isnan(labels).any(dim=1)
        valid_labels = labels[mask]
        valid_labels = valid_labels.to(torch.int64)
        
        class_indices = (valid_labels[:, 0] * 4 + valid_labels[:, 1] * 2 + valid_labels[:, 2]).long()
        data_y = class_indices.unsqueeze(1)
        # print('data.y', data.y)
        
        # new_data_list.append(data)
    # new_dataset = dataset.__class__(new_data_list)
    
    return data_y

def compute_node_feature_variances(dataset, k):
    all_variances = []

    for data in dataset:
        # x: [num_nodes, num_features]
        x = data.x
        x = x.float()
        variances = torch.var(x, dim=0)
        all_variances.append(variances)

    global_variances = torch.mean(torch.stack(all_variances), dim=0)
    _, top_k_indices = torch.topk(global_variances, k)
    # print('top_k_indices', top_k_indices)

    return top_k_indices



def split_dataset_ood_edge(dataset, edge_type, threshold=0.25, swap_prob=0., target_ratio=0.8):
    train_graphs, test_graphs = [], []

    for graph_idx in range(len(dataset)):
        edge_attr = dataset[graph_idx].edge_attr
        total_edges = edge_attr.size(0)

        if total_edges == 0:
            train_graphs.append(graph_idx)
        else:
            edge_type_ratio = (edge_attr [:, edge_type] == 1).float().mean().item()
            # print('edge_type_ratio', edge_type_ratio)
        
            if edge_type_ratio < threshold:
                train_graphs.append(graph_idx)
            else:
                test_graphs.append(graph_idx)

    train_size = len(train_graphs)
    test_size = len(test_graphs)
    
    total_size = len(dataset)
    target_train_size = int(total_size * target_ratio)
    target_test_size = total_size - target_train_size
    
    if train_size > target_train_size:
        swap_size = train_size - target_train_size
        swap_indices = torch.randperm(train_size)[:swap_size]
        for idx in swap_indices:
            test_graphs.append(train_graphs[idx])
        train_graphs = [graph for i, graph in enumerate(train_graphs) if i not in swap_indices]

    elif test_size > target_test_size:
        swap_size = test_size - target_test_size
        swap_indices = torch.randperm(test_size)[:swap_size]
        for idx in swap_indices:
            train_graphs.append(test_graphs[idx])
        test_graphs = [graph for i, graph in enumerate(test_graphs) if i not in swap_indices]


    while len(train_graphs) + len(test_graphs) < total_size:
        if len(train_graphs) < target_train_size:
            train_graphs.append(test_graphs.pop())
        else:
            test_graphs.append(train_graphs.pop())
    
    if swap_prob > 0:
        swap_size = int(swap_prob * min(len(train_graphs), len(test_graphs)))
        if swap_size > 0:
            swap_train = train_graphs[:swap_size]
            swap_test = test_graphs[:swap_size]
            train_graphs[:swap_size], test_graphs[:swap_size] = swap_test, swap_train
    
    val_size = int(len(train_graphs) / 8)
    final_train_graphs, val_graphs = train_test_split(train_graphs, test_size=val_size, random_state=12345)
    
    return final_train_graphs, val_graphs, test_graphs


def split_dataset_ood_node(dataset, train_threshold=0.2, test_threshold=0.5, swap_prob=0.1, target_ratio=0.8):
    top_k_indices = compute_node_feature_variances(dataset, 2)
    train_graphs, test_graphs = [], []

    for graph_idx in range(len(dataset)):
        node_attr = dataset[graph_idx].x
        total_nodes = node_attr.size(0)

        if total_nodes == 0:
            train_graphs.append(graph_idx)
        else:
            feature_0_proportion = (node_attr[:, top_k_indices[0]] == 1).float().mean().item()
            feature_1_proportion = (node_attr[:, top_k_indices[1]] == 1).float().mean().item()
        
            if feature_0_proportion < train_threshold and feature_1_proportion > test_threshold:
                test_graphs.append(graph_idx)
            else:
                train_graphs.append(graph_idx)

    train_size = len(train_graphs)
    test_size = len(test_graphs)
    
    total_size = len(dataset)
    target_train_size = int(total_size * target_ratio)
    target_test_size = total_size - target_train_size
    
    if train_size > target_train_size:
        swap_size = train_size - target_train_size
        swap_indices = torch.randperm(train_size)[:swap_size]
        for idx in swap_indices:
            test_graphs.append(train_graphs[idx])
        train_graphs = [graph for i, graph in enumerate(train_graphs) if i not in swap_indices]

    elif test_size > target_test_size:
        swap_size = test_size - target_test_size
        swap_indices = torch.randperm(test_size)[:swap_size]
        for idx in swap_indices:
            train_graphs.append(test_graphs[idx])
        test_graphs = [graph for i, graph in enumerate(test_graphs) if i not in swap_indices]

    while len(train_graphs) + len(test_graphs) < total_size:
        if len(train_graphs) < target_train_size:
            train_graphs.append(test_graphs.pop())
        else:
            test_graphs.append(train_graphs.pop())
    
    if swap_prob > 0:
        swap_size = int(swap_prob * min(len(train_graphs), len(test_graphs)))
        if swap_size > 0:
            swap_train = train_graphs[:swap_size]
            swap_test = test_graphs[:swap_size]
            train_graphs[:swap_size], test_graphs[:swap_size] = swap_test, swap_train
    
    val_size = int(len(train_graphs) / 8)
    final_train_indices, val_indices = train_test_split(train_graphs, test_size=val_size, random_state=12345)
    
    return final_train_indices, val_indices, test_graphs


def split_dataset_ood_label(dataset, train_threshold=0.2, test_threshold=0.8, swap_prob=0.1, target_ratio=0.8, scale_factor=5):
    train_graphs = []
    test_graphs = []

    for graph_idx in range(len(dataset)):
        label = dataset[graph_idx].y.item()
        if label == 0:
            train_graphs.append(graph_idx)
        else:
            test_graphs.append(graph_idx)

    train_size = len(train_graphs)
    test_size = len(test_graphs)
    total_size = len(dataset)
    # print('train_size', train_size)
    # print('test_size', test_size)

    if train_size > test_size * scale_factor:
        desired_train_size = test_size * scale_factor
        total_size = desired_train_size + test_size
        if train_size > desired_train_size:
            train_graphs = train_graphs[:desired_train_size]

    target_train_size = int(total_size * target_ratio)
    target_test_size = total_size - target_train_size
    new_train_size = len(train_graphs)
    # print('target_train_size', target_train_size)
    # print('target_test_size', target_test_size)

  
    if new_train_size > target_train_size:
        swap_size = new_train_size - target_train_size
        swap_indices = torch.randperm(new_train_size)[:swap_size]
        for idx in swap_indices:
            test_graphs.append(train_graphs[idx])
        train_graphs = [graph for i, graph in enumerate(train_graphs) if i not in swap_indices]

    elif test_size > target_test_size:
        swap_size = test_size - target_test_size
        swap_indices = torch.randperm(test_size)[:swap_size]
        for idx in swap_indices:
            train_graphs.append(test_graphs[idx])
        test_graphs = [graph for i, graph in enumerate(test_graphs) if i not in swap_indices]

    while len(train_graphs) + len(test_graphs) < total_size:
        if len(train_graphs) < target_train_size:
            train_graphs.append(test_graphs.pop())
        else:
            test_graphs.append(train_graphs.pop())
    
    while len(train_graphs) + len(test_graphs) < total_size:
        if len(train_graphs) < target_train_size:
            if test_graphs:  
                train_graphs.append(test_graphs.pop())
        else:
            if train_graphs:  
                test_graphs.append(train_graphs.pop())

    if swap_prob > 0:
        swap_size = int(swap_prob * min(len(train_graphs), len(test_graphs)))
        if swap_size > 0:
            swap_train = train_graphs[:swap_size]
            swap_test = test_graphs[:swap_size]
            train_graphs[:swap_size], test_graphs[:swap_size] = swap_test, swap_train

    
    val_size = int(len(train_graphs) / 8)
    final_train_indices, val_indices = train_test_split(train_graphs, test_size=val_size, random_state=12345)
    
    return final_train_indices, val_indices, test_graphs




def split_ogb_bias(dataset, bias=0.5):
    all_indices = list(range(len(dataset)))
    label_0_indices = [idx for idx in all_indices if dataset[idx].y.item() == 0]
    label_1_indices = [idx for idx in all_indices if dataset[idx].y.item() == 1]
    # print('len(label_0_indices):', len(label_0_indices))
    # print('len(label_1_indices):', len(label_1_indices))
    
    num_label_1_test = min(len(label_1_indices), int(len(label_1_indices) / 8.2))
    num_label_0_test = num_label_1_test

    test_label_0_indices = label_0_indices[:num_label_0_test]
    test_label_1_indices = label_1_indices[:num_label_1_test]
    
    # test_label_0_indices = np.random.choice(label_0_indices, num_label_0_test, replace=False)
    # test_label_1_indices = np.random.choice(label_1_indices, num_label_1_test, replace=False)
    # print('len(test_label_1_indices):', len(test_label_1_indices))
    
    test_indices = list(test_label_0_indices) + list(test_label_1_indices)
    test_size = len(test_indices)
    test_set = set(test_indices)  
    
    remaining_indices = [idx for idx in all_indices if idx not in test_set]
    # print('remaining_data:', len(dataset)-test_size)
    # print('len(remaining_indices):', len(remaining_indices))
    remaining_label_0_indices = [idx for idx in remaining_indices if dataset[idx].y.item() == 0]
    remaining_label_1_indices = [idx for idx in remaining_indices if dataset[idx].y.item() == 1]
    
 
    train_size = int(test_size * 4)
    num_label_0_train = int(train_size * (1 - bias))
    num_label_1_train = train_size - num_label_0_train
    
    # train_label_0_indices = np.random.choice(remaining_label_0_indices, num_label_0_train, replace=False)
    # train_label_1_indices = np.random.choice(remaining_label_1_indices, num_label_1_train, replace=False)
    train_label_0_indices = remaining_label_0_indices[:num_label_0_train]
    train_label_1_indices = remaining_label_1_indices[:num_label_1_train]
    
    # print('len(train_label_0_indices):', len(train_label_0_indices))
    # print('len(train_label_1_indices):', len(train_label_1_indices))
    
    train_indices = list(train_label_0_indices) + list(train_label_1_indices)
    # train_set = set(train_indices)  # Convert to set for faster lookups
    
    val_size = int(train_size / 8)
    final_train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=12345)
    # print('total size:', len(final_train_indices)+len(val_indices)+len(test_indices))
    
    return final_train_indices, val_indices, test_indices


