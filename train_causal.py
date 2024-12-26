import os
import csv
import logging
from datetime import datetime
import json 
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
from utils import k_fold, num_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_real_ood(train_set, val_set, test_set, model_func=None, args=None):
    exp_dir = os.path.join('exp', f"{args.model}_{args.dataset}_{args.spliting}_{args.ablation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True)
    
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
        
    model = model_func(train_set.num_features, train_set.num_edge_features, train_set.num_classes).to(device)
    # print('model:',model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
    
    best_val_acc, update_test_acc_co, update_test_acc_c, update_test_acc_o, update_epoch = 0, 0, 0, 0, 0

 
    train_csv = open(os.path.join(exp_dir, 'train_results.csv'), 'w', newline='')
    val_csv = open(os.path.join(exp_dir, 'val_results.csv'), 'w', newline='')
    test_csv = open(os.path.join(exp_dir, 'test_results.csv'), 'w', newline='')
    
    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)
    test_writer = csv.writer(test_csv)
    
    train_writer.writerow(['epoch', 'train_loss', 'loss_c', 'loss_o', 'loss_co', 'train_acc_o'])
    val_writer.writerow(['epoch', 'val_acc_co', 'val_acc_c', 'val_acc_o'])
    test_writer.writerow(['epoch', 'test_acc_co', 'test_acc_c', 'test_acc_o'])

    for epoch in range(1, args.epochs + 1):
        # print('epoch:',epoch)
        if args.ablation == "remove_kl":
            # print('remove_kl training...')
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_remove_kl_epoch(model, optimizer, train_loader, device, args)
            val_acc_co, val_acc_c, val_acc_o = eval_acc_remove_kl(model, val_loader, device, args)
            test_acc_co, test_acc_c, test_acc_o = eval_acc_remove_kl(model, test_loader, device, args)
        elif args.ablation == "remove_co":
            # print('remove_co training...')
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_remove_co_epoch(model, optimizer, train_loader, device, args)
            val_acc_co, val_acc_c, val_acc_o = eval_acc_remove_co(model, val_loader, device, args)
            test_acc_co, test_acc_c, test_acc_o = eval_acc_remove_co(model, test_loader, device, args)
        elif args.ablation == "remove_all":
            # print('remove_all training...')
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_remove_all_epoch(model, optimizer, train_loader, device, args)
            val_acc_co, val_acc_c, val_acc_o = eval_acc_remove_all(model, val_loader, device, args)
            test_acc_co, test_acc_c, test_acc_o = eval_acc_remove_all(model, test_loader, device, args)
        elif args.ablation == "none":
            # print('causal training...')
            train_loss, loss_c, loss_o, loss_co, train_acc_o = train_causal_epoch(model, optimizer, train_loader, device, args)
            val_acc_co, val_acc_c, val_acc_o = eval_acc_causal(model, val_loader, device, args)
            test_acc_co, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)
        else:
            assert False, f"Unknown ablation type: {args.ablation}"

        lr_scheduler.step()
        
        train_writer.writerow([epoch, train_loss, loss_c, loss_o, loss_co, train_acc_o])
        val_writer.writerow([epoch, val_acc_co, val_acc_c, val_acc_o])
        test_writer.writerow([epoch, test_acc_co, test_acc_c, test_acc_o])

        if val_acc_o > best_val_acc:
            best_val_acc = val_acc_o
            update_test_acc_co = test_acc_co
            update_test_acc_c = test_acc_c
            update_test_acc_o = test_acc_o
            update_epoch = epoch
            
            # Save the best model
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

        print("Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Update Test:[co:{:.2f},c:{:.2f},o:{:.2f}] at Epoch:[{}] | lr:{:.6f}"
                .format(args.model,
                        epoch, 
                        args.epochs,
                        train_loss,
                        loss_c,
                        loss_o,
                        loss_co,
                        train_acc_o * 100, 
                        val_acc_o * 100,
                        test_acc_o * 100, 
                        update_test_acc_co * 100,
                        update_test_acc_c * 100,  
                        update_test_acc_o * 100, 
                        update_epoch,
                        optimizer.param_groups[0]['lr']))

    train_csv.close()
    val_csv.close()
    test_csv.close()

    log_data = {
        'model': args.model,
        'dataset': args.dataset,
        'spliting': args.spliting,
        'ablation': args.ablation,
        'swap_prob': args.swap_prob,
        'scale_factor': args.scale_factor,
        'edge_replace_rate': args.replace,
        'bias': args.bias,
        "best_val_acc": best_val_acc,
        "update_test_acc_co": update_test_acc_co,
        "update_test_acc_c": update_test_acc_c,
        "update_test_acc_o": update_test_acc_o,
        "update_epoch": update_epoch
    }

    with open(os.path.join(exp_dir, 'log.json'), 'w') as f:
        json.dump(log_data, f, indent=4)

def train_causal_epoch(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):

        # print('it',it)
        
        optimizer.zero_grad()
        data = data.to(device)
        
        one_hot_target = data.y.view(-1)  
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random, replacement_ratio=args.replace)
 
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        loss = args.c * c_loss + args.o * o_loss + args.co * co_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

def train_remove_kl_epoch(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    # total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):

        # print('it',it)
        
        optimizer.zero_grad()
        data = data.to(device)
        
        one_hot_target = data.y.view(-1)   
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random, replacement_ratio=args.replace)
 
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        # c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        co_loss = F.nll_loss(co_logs, one_hot_target)
        # print('co_logs',co_logs)
        # print('co_loss',co_loss)
        loss = args.o * o_loss + args.co * co_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        # total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = 0
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

def train_remove_co_epoch(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    # total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):

        # print('it',it)
        
        optimizer.zero_grad()
        data = data.to(device)
        
        one_hot_target = data.y.view(-1)   
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random, replacement_ratio=args.replace)
 
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        # co_loss = F.nll_loss(co_logs, one_hot_target)

        loss = args.o * o_loss + args.c * c_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        # total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_co = 0
    total_loss_o = total_loss_o / num
    total_loss_c = total_loss_c / num
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

def train_remove_all_epoch(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    # total_loss_c = 0
    total_loss_o = 0
    # total_loss_co = 0
    correct_o = 0
    
    for it, data in enumerate(loader):

        # print('it',it)
        
        optimizer.zero_grad()
        data = data.to(device)
        
        one_hot_target = data.y.view(-1)   
        c_logs, o_logs, co_logs = model(data, eval_random=args.with_random, replacement_ratio=args.replace)
 
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        
        # c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target)
        # co_loss = F.nll_loss(co_logs, one_hot_target)

        loss = o_loss

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        # total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        # total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_co = 0
    total_loss_o = total_loss_o / num
    total_loss_c = 0
    correct_o = correct_o / num
    return total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o

def eval_acc_causal(model, loader, device, args):
    
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

def eval_acc_remove_kl(model, loader, device, args):
    
    model.eval()
    eval_random = args.eval_random
    correct = 0
    # correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            # pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        correct += pred.eq(data.y.view(-1)).sum().item()
        # correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = 0
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

def eval_acc_remove_co(model, loader, device, args):
    
    model.eval()
    eval_random = args.eval_random
    # correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            # pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        # correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = 0
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o

def eval_acc_remove_all(model, loader, device, args):
    
    model.eval()
    eval_random = args.eval_random
    # correct = 0
    # correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            # pred = co_logs.max(1)[1]
            # pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        # correct += pred.eq(data.y.view(-1)).sum().item()
        # correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = 0
    acc_c = 0
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o