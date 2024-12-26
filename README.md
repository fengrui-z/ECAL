# Catch Causal Signals from Edges for Label Imbalance in Graph Classification [ICASSP2025]


This repository contains the implementation of our proposed **Edge-Enhanced Causal Attention Learning (ECAL)** framework, which effectively incorporates edge features into causal attention mechanisms to tackle **label imbalance** in graph classification tasks. Our method disentangles causal subgraphs from original graphs and reshapes graph representations using edge features, leading to improved performance on real-world datasets.

---

## Table of Contents
- [Catch Causal Signals from Edges for Label Imbalance in Graph Classification \[ICASSP2025\]](#catch-causal-signals-from-edges-for-label-imbalance-in-graph-classification-icassp2025)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [A quick training example is](#a-quick-training-example-is)
  - [Handling Label Imbalance](#handling-label-imbalance)
  - [Citation](#citation)

---

## Introduction

Graph neural networks (GNNs) have achieved significant success in graph classification tasks. However, real-world datasets often face **label imbalance**, making it challenging for models to generalize effectively. While causal discovery and causal attention mechanisms have been proposed to address this, most existing methods **overlook the importance of edge features** in graphs.

In this work, we propose **ECAL**, a framework that enhances causal attention mechanisms by incorporating edge features:
- **Edge-Featured Graph Attention Networks (EGAT)** modules are used to capture causal signals from edges.
- The framework splits graphs into **causal** and **trivial** subgraphs and reshapes representations to improve classification performance under label imbalance.
- Extensive experiments demonstrate the effectiveness of ECAL on benchmark datasets.

---


## Dependencies

Ensure the following dependencies are installed:

- Python >= 3.8
- PyTorch >= 1.10
- torch_geometric

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## A quick training example is

```bash
python main_real.py --model ECALv2 --dataset PTC_MM --epochs 200

```

---

## Handling Label Imbalance

We provide methods to create imbalanced datasets for graph classification. These methods allow users to simulate real-world scenarios where certain classes are underrepresented. For datasets with label imbalance, set the argument **--spliting** to label to configure the model for imbalanced data. For example:

```bash
python main_real.py --model GCN --dataset PTC_MM --epochs 200 --spliting label --scale_factor 2 --swap_prob 0.2
```
Here,
- **--swap_prob** controls the proportion of data swapped between train/test sets.
- **--scale_factor** limits the maximum ratio of train set size to test set size.

---
## Citation
```bash
@inproceedings{zhang2025ecal,
  title={Catch Causal Signals from Edges for Label Imbalance in Graph Classification},
  author={Fengrui Zhang and Yujia Yin and Hongzong Li and Yifan Chen and Tianyi Qu},
  booktitle={Proceedings of the 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year={2025},
  publisher={IEEE}
}
```

For any questions or issues, please contact yifanc@hkbu.edu.hk or qutianyi@sf-express.com.