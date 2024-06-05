import os.path as osp
import re
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from feature_expansion import FeatureExpander
from train_causal import train_causal_real
import opts
import warnings
warnings.filterwarnings('ignore')
import time

# Function to load Citeseer data
def load_citeseer_data(root='data/Citeseer'):
    dataset = Planetoid(root=root, name='Citeseer')
    return dataset

# Function to preprocess Citeseer data
def preprocess_citeseer_data(dataset, feat_str="deg+ak3+reall"):
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall(r"odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall(r"an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    centrality = feat_str.find("cent") >= 0

    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k, centrality=centrality).transform

    for data in dataset:
        data = pre_transform(data)

    return dataset

# Main function
def main():
    args = opts.parse_args()
    dataset_name = "Citeseer"
    dataset = load_citeseer_data()
    dataset = preprocess_citeseer_data(dataset)
    model_func = opts.get_model(args)
    train_causal_real(dataset, model_func, args)

if __name__ == '__main__':
    main()

