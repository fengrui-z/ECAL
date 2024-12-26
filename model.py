from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GATConv
from edge_gcn import EGATGCNConv, EGAT, GCNConv
import random
import pdb

torch.set_printoptions(threshold=10000, edgeitems=3, linewidth=200)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ECausalGCN(torch.nn.Module):
    """ECALv2."""
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       heads=3, layers=3):
        super(ECausalGCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        EGATConv = partial(EGAT, heads=3, layers=3)

        hidden_in_node = num_features
        hidden_in_edge = num_edge_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in_node)
        # print('hidden',hidden)
        self.conv_feat = EGAT(hidden_in_node, hidden_in_edge, hidden, heads, layers) 
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(EGATConv(hidden, hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        self.context_convs = EGATConv(hidden, hidden, hidden)
        self.objects_convs = EGATConv(hidden, hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def generate_random_one_hot(self, num_edges, num_classes):
        random_edge_attr = torch.zeros(num_edges, num_classes).to(device)
        for i in range(num_edges):
            random_class = random.randint(0, num_classes - 1)
            random_edge_attr[i][random_class] = 1
        return random_edge_attr

    def forward(self, data, eval_random=True, replacement_ratio=0.):
        x = data.x if data.x is not None else data.feat
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()

        num_edges = edge_attr.size(0)
        num_classes = edge_attr.size(1)
        num_edges_to_replace = int(replacement_ratio * num_edges)

        indices_to_replace = random.sample(range(num_edges), num_edges_to_replace)

        new_edge_attr = self.generate_random_one_hot(num_edges_to_replace, num_classes)

        edge_attr[indices_to_replace] = new_edge_attr


        row, col = edge_index
        # print('before x',x[0])
        x = self.bn_feat(x)
        
        x, edge, _ = self.conv_feat(x, edge_index, edge_attr)
        # print('x_after',x[0])
        x = F.relu(x) 
        edge = F.relu(edge)  
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            
            x, edge, edge_attn = conv(x, edge_index, edge)
          
            x = F.relu(x)  
            edge = F.relu(edge)
        
        edge_weight_c = edge_attn
        edge_weight_o = 1 - edge_attn
        edge_c = edge_weight_c.view(-1, 1) * edge
        edge_o = edge_weight_o.view(-1, 1) * edge

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        # print('node_att',node_att.shape)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x

        xc, _, _ = self.context_convs(self.bnc(xc), edge_index, self.bnc(edge_c))
        xo, _, _ = self.objects_convs(self.bno(xo), edge_index, self.bno(edge_o))
        # if torch.isnan(xc).any():
        #     print('the xc has nan')
        xc = F.relu(xc)
        xo = F.relu(xo)


        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        # print('xc',xc.shape)

        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        x = self.fc1_bn_c(x)
        
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class ECausalGAT(torch.nn.Module):
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, 
                       args, 
                       head=4, 
                       dropout=0.2,
                       heads=3, layers=3):
        super(ECausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        EGATConv = partial(EGAT, heads=3, layers=3)  

        hidden_in_node = num_features
        hidden_in_edge = num_edge_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in_node)
        self.conv_feat = EGAT(hidden_in_node, hidden_in_edge, hidden, heads, layers)
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout, edge_dim = hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        self.context_convs = EGATConv(hidden, hidden, hidden)
        self.objects_convs = EGATConv(hidden, hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def generate_random_one_hot(self, num_edges, num_classes):
        random_edge_attr = torch.zeros(num_edges, num_classes)
        for i in range(num_edges):
            random_class = random.randint(0, num_classes - 1)
            random_edge_attr[i][random_class] = 1
        return random_edge_attr


    def forward(self, data, eval_random=True, replacement_ratio=0.):
        x = data.x if data.x is not None else data.feat
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()

        num_edges = edge_attr.size(0)
        num_classes = edge_attr.size(1)
        num_edges_to_replace = int(replacement_ratio * num_edges)

        indices_to_replace = random.sample(range(num_edges), num_edges_to_replace)

        new_edge_attr = self.generate_random_one_hot(num_edges_to_replace, num_classes)

        edge_attr[indices_to_replace] = new_edge_attr


        row, col = edge_index
        x = self.bn_feat(x)
        x, edge, edge_attn = self.conv_feat(x, edge_index, edge_attr)
        x = F.relu(x)  
        edge = F.relu(edge)
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = conv(x, edge_index, edge)
            x = F.relu(x)

        edge_weight_c = edge_attn
        edge_weight_o = 1 - edge_attn
        edge_c = edge_weight_c.view(-1, 1) * edge
        edge_o = edge_weight_o.view(-1, 1) * edge

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc, _, _ = self.context_convs(self.bnc(xc), edge_index, self.bnc(edge_c))
        xo, _, _ = self.objects_convs(self.bno(xo), edge_index, self.bno(edge_o))
        xc = F.relu(xc)
        xo = F.relu(xo)

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class EGCNNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, args, 
                       num_feat_layers=1, 
                       num_conv_layers=3,
                       num_fc_layers=2, gfn=False, collapse=False, residual=False,
                       res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                       edge_norm=True):
        super(EGCNNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(EGATGCNConv, edge_norm=edge_norm, gfn=gfn)  # 使用EGATGCNConv

        hidden_in_node = num_features
        hidden_in_edge = num_edge_features
        self.bn_feat = BatchNorm1d(hidden_in_node)
        hidden = args.hidden
        # print('hidden',hidden)
        self.conv_feat = EGATGCNConv(hidden_in_node, hidden_in_edge, hidden, gfn=True)  # 使用EGATGCNConv
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()
        x = self.bn_feat(x)
        x, edge, _ = self.conv_feat(x, edge_index, edge_attr)
        x = F.relu(x)
        edge = F.relu(edge)
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x, _, _ = conv(x, edge_index, edge)
            x = F.relu(x)
            # edge = F.relu(edge)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
   
    
class EGATNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, args, 
                       num_feat_layers=1, 
                       num_conv_layers=3,
                       num_fc_layers=2, gfn=False, collapse=False, residual=False,
                       res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                       edge_norm=True,
                       heads=3, layers=3):
        super(EGATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(EGAT, heads=3, layers=3) 

        hidden_in_node = num_features
        hidden_in_edge = num_edge_features
        hidden = args.hidden
        self.bn_feat = BatchNorm1d(hidden_in_node)
        # print('hidden',hidden)
        self.conv_feat = EGAT(hidden_in_node, hidden_in_edge, hidden, heads, layers)  
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()
        x = self.bn_feat(x)
        x, edge, _ = self.conv_feat(x, edge_index, edge_attr)
        x = F.relu(x)
        edge = F.relu(edge)
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x, _, _ = conv(x, edge_index, edge)
            x = F.relu(x)
            # edge = F.relu(edge)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)


class GCNNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, hidden, 
                       num_feat_layers=1, 
                       num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(GCNNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        x = x.float()
        edge_index, batch = data.edge_index, data.batch
        # print('beforex',x)
        x = self.bn_feat(x)
        # print('afterx',x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)


class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_edge_features,
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):
        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = EGAT(hidden_in, num_edge_features, hidden)  
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout,edge_dim=hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        x = x.float()
        edge_attr = edge_attr.float()
        x = self.bn_feat(x)
        x, edge, _ = self.conv_feat(x, edge_index, edge_attr)
        x = F.relu(x)
        edge = F.relu(edge)
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index, edge))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
    

class GATNetv1(torch.nn.Module):
    def __init__(self, num_features, 
                       num_edge_features,
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNetv1, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x = x.float()
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)



class CausalGCN(torch.nn.Module):
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0, 
                       edge_norm=True):
        super(CausalGCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.with_random = args.with_random
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, replacement_ratio=0.):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x = x.float()
        # edge_attr = edge_attr.float()
        row, col = edge_index
        # print('x_before',x)
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)

        if self.without_edge_attention:
            edge_att = 0.5 * torch.ones(edge_rep.shape[0], 2).cuda()
        else:
            edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        if self.without_node_attention:
            node_att = 0.5 * torch.ones(x.shape[0], 2).cuda()
        else:
            node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)

        return xc_logis, xo_logis, xco_logis


    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if self.with_random:
            if eval_random:
                random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class CausalGAT(torch.nn.Module):
    def __init__(self, num_features,
                       num_edge_features,
                       num_classes, 
                       args, 
                       head=4, 
                       dropout=0.2):
        super(CausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, replacement_ratio=0.):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x = x.float()
        # edge_attr = edge_attr.float()
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis
    
