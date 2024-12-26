import argparse
from model import ECausalGCN, ECausalGAT, EGCNNet, GATNet, EGATNet, GCNNet, CausalGCN, CausalGAT, GATNetv1
import numpy as np
import random
import torch
from itertools import product

def parse_args():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    #################### toy example #######################
    parser.add_argument('--pretrain', type=int, default=30)
    parser.add_argument('--data_num', type=int, default=2000)
    parser.add_argument('--node_num', type=int, default=15)
    parser.add_argument('--max_degree', type=int, default=10)
    parser.add_argument('--feature_dim', type=int, default=-1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--shape_num', type=int, default=1)
    parser.add_argument('--bias', type=float, default=0.7)
    parser.add_argument('--penalty_weight', default=0.1, type=float, help='penalty weight')
    parser.add_argument('--train_type', type=str, default="base", help="irm, dro, base")

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--the', type=int, default=0)
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--eval_random', type=str2bool, default=False)
    parser.add_argument('--normalize', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--inference', type=str2bool, default=False)
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    
    parser.add_argument('--k', type=int, default=3)
    #################### Causal GNN settings #######################
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--o', type=float, default=1.0)
    parser.add_argument('--co', type=float, default=0.5)
    parser.add_argument('--harf_hidden', type=float, default=0.5)
    parser.add_argument('--cat_or_add', type=str, default="add")
    ##################### baseline training ######################
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--fc_num', type=str, default="222")
    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--swap_prob', type=float, default=0.2)
    parser.add_argument('--replace', type=float, default=0.)
    parser.add_argument('--spliting', type=str, default="label", choices=["edge", "node", "label", "ogb_bias"], 
                        help="Method to split the dataset (default: edge). Options: 'edge', 'node', 'label'")
    parser.add_argument('--ablation', type=str, default="none", choices=["remove_kl", "remove_co", "remove_all", "none"], help="")
    parser.add_argument('--edge_type', type=int, default=1, help="Edge type for edge spliting (default: 1)")
    parser.add_argument('--save_dir', type=str, default="debug")
    parser.add_argument('--dataset', type=str, default="PTC_MR")
    parser.add_argument('--epoch_select', type=str, default='test_max')
    parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN, GAT, CausalGCN, CausalGIN, CausalGAT")
    parser.add_argument('--hidden', type=int, default=128)

    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--global_pool', type=str, default="sum")
    args = parser.parse_args()
    print_args(args)
    setup_seed(args.seed)
    return args

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def setup_seed(seed):
    # print('seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args):

    def model_func1(num_features, num_edge_features, num_classes):
        return GCNNet(num_features, num_edge_features, num_classes, args.hidden) 
    
    def model_func2(num_features, num_edge_features, num_classes):
        return GATNet(num_features, num_edge_features, num_classes, args.hidden) 
    
    def model_func3(num_features, num_edge_features, num_classes):
        return EGCNNet(num_features, num_edge_features, num_classes, args)

    def model_func4(num_features, num_edge_features, num_classes):
        return EGATNet(num_features, num_edge_features, num_classes, args)

    def model_func5(num_features,num_edge_features, num_classes):
        return ECausalGCN(num_features, num_edge_features, num_classes, args) 

    def model_func6(num_features, num_edge_features, num_classes):
        return ECausalGAT(num_features, num_edge_features, num_classes, args) 

    def model_func7(num_features, num_edge_features, num_classes):
        return CausalGCN(num_features, num_edge_features, num_classes, args) 
    
    def model_func8(num_features, num_edge_features, num_classes):
        return CausalGAT(num_features, num_edge_features, num_classes, args)
    
    def model_func9(num_features, num_edge_features, num_classes):
        return GATNetv1(num_features, num_edge_features, num_classes, args.hidden)



    if args.model == "GCN":
        model_func = model_func1
    elif args.model == "EGATv1":
        model_func = model_func2
    elif args.model == "EGCN":
        model_func = model_func3
    elif args.model == "EGATv2":
        model_func = model_func4
    elif args.model == "ECALv2":
        model_func = model_func5
    elif args.model == "ECALv1":
        model_func = model_func6
    elif args.model == "CALGCN":
        model_func = model_func7
    elif args.model == "CALGAT":
        model_func = model_func8
    elif args.model == "GAT":
        model_func = model_func9

    else:
        assert False
    return model_func

