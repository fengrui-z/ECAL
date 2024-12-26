from train import train_baseline_syn
from train_causal import train_real_ood
import opts
import warnings
warnings.filterwarnings('ignore')
import time
import torch
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
import utils


def main():
    torch.use_deterministic_algorithms(True)
    args = opts.parse_args()
    

    if "ogbg" in args.dataset:
        dataset = PygGraphPropPredDataset(name=args.dataset, root=args.data_root)
    else:
        dataset = TUDataset(root=args.data_root, name=args.dataset, use_node_attr=True, use_edge_attr=True)



    if args.spliting == "ogb_bias":
        if args.dataset != "ogbg-molhiv":
            print(f"Error: For split type 'ogb_bias', dataset must be 'ogbg-molhiv'. Provided dataset is {args.dataset}.")
            return
        train_indices, val_indices, test_indices = utils.split_ogb_bias(dataset, bias=args.bias)
    elif args.spliting == "edge":
        train_indices, val_indices, test_indices = utils.split_dataset_ood_edge(dataset, args.edge_type, threshold=0.25, swap_prob=0., target_ratio=0.8)
    elif args.spliting == "node":
        train_indices, val_indices, test_indices = utils.split_dataset_ood_node(dataset, train_threshold=0.6, test_threshold=0.1, swap_prob=0., target_ratio=0.8)
    elif args.spliting == "label":
        train_indices, val_indices, test_indices = utils.split_dataset_ood_label(dataset, train_threshold=0.6, test_threshold=0.1, swap_prob=args.swap_prob, target_ratio=0.8, scale_factor=args.scale_factor)
    else:
        assert False, f"Unknown splitting type: {args.spliting}"

    if not (train_indices and val_indices and test_indices):
        print(f"Skipping training for dataset {args.dataset} and splitting type {args.spliting}")
        return

    train_set = dataset[train_indices]
    val_set = dataset[val_indices]
    test_set = dataset[test_indices]

    if args.model in ["GCN", "GAT", "EGATv1", "EGATv2"]:
        model_func = opts.get_model(args)
        train_baseline_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    elif args.model in ["ECALv1", "ECALv2", "CALGCN", "CALGAT"]:
        model_func = opts.get_model(args)
        train_real_ood(train_set, val_set, test_set, model_func=model_func, args=args)
    else:
        assert False, f"Unknown model type: {args.model}"

if __name__ == '__main__':
    main()
