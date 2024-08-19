from datasets import get_dataset
from train import train_baseline_syn
from train_causal import train_causal_real
import opts
import warnings
warnings.filterwarnings('ignore')
import time

def main():
    # 解析参数
    args = opts.parse_args()
    
    # 根据参数选择并加载数据集
    dataset_name, feat_str, _ = opts.create_n_filter_triples([args.dataset])[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
    
    # 获取模型
    model_func = opts.get_model(args)
    
    # 训练模型
    train_causal_real(dataset, model_func, args)
    
if __name__ == '__main__':
    main()
