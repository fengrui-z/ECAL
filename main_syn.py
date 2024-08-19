from train import train_baseline_syn
from train_causal import train_causal_syn
from opts import setup_seed
import torch
import opts
import os
import utils
import warnings
warnings.filterwarnings('ignore')

def main():
    # Parse command-line arguments or config file parameters
    args = opts.parse_args()
    save_path = "data"
    os.makedirs(save_path, exist_ok=True)

    # Load the synthetic dataset from disk or generate it if not available
    try:
        dataset = torch.load(save_path + "/syn_dataset.pt")
    except:
        dataset = utils.graph_dataset_generate(args, save_path)

    # Split the dataset into training, validation, and test sets
    train_set, val_set, test_set, the = utils.dataset_bias_split(dataset, args, bias=args.bias, split=[7, 1, 2], total=args.data_num * 4)

    # Print dataset information such as number of nodes, edges, and class distribution
    group_counts = utils.print_dataset_info(train_set, val_set, test_set, the)

    # Select the appropriate model and training function based on the specified model type
    if args.model in ["GIN", "GCN", "GAT"]:
        model_func = opts.get_model(args)
        train_baseline_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        model_func = opts.get_model(args)
        train_causal_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    else:
        assert False, f"Unknown model type: {args.model}"

if __name__ == '__main__':
    main()
