import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout=0.):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EGAT(MessagePassing):
    def __init__(self, in_node_features, in_edge_features, out_features, heads=3, layers=3, bias=True, 
                 return_attn_weights=True, use_attention_weights=True, negative_slope=0.2, dropout=0.1, **kwargs):
        super(EGAT, self).__init__(node_dim=0, **kwargs)

        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.out_features = out_features
        self.heads = heads
        self.return_attn_weights = return_attn_weights
        self.use_attention_weights = use_attention_weights
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        
        self.node_transform = nn.Linear(in_node_features, out_features, bias=True)
        self.edge_transform = nn.Linear(in_edge_features, out_features, bias=True)
        self.node_transform_i = nn.Linear(in_node_features, heads * out_features, bias=False)
        self.node_transform_j = nn.Linear(in_node_features, heads * out_features, bias=False)
        self.edge_transform_ij = nn.Linear(in_edge_features, heads * out_features, bias=False)

        
        self.attn_mlp = MLP(3 * heads * out_features, heads * out_features, heads * out_features, layers)

        
        self.attn_param = Parameter(torch.FloatTensor(size=(1, heads, out_features)))

       
        self.node_mlp = MLP(heads * out_features, out_features, out_features, layers)
        self.edge_mlp = MLP(heads * out_features, out_features, out_features, layers)

        self.linear = nn.Linear(heads, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.node_transform.weight)
        glorot(self.edge_transform.weight)
        glorot(self.node_transform_i.weight)
        glorot(self.node_transform_j.weight)
        glorot(self.edge_transform_ij.weight)
        glorot(self.attn_param)

    def forward(self, node_features, edge_index, edge_features, size=None):
        H, C = self.heads, self.out_features
        
        node_feat_i = self.node_transform_i(node_features)  # shape [N, H*C]
        node_feat_j = self.node_transform_j(node_features)  # shape [N, H*C]
        # print(self.edge_transform_ij.weight.shape)
        edge_feat_ij = self.edge_transform_ij(edge_features)  # shape [E, H*C]
        # print('node_features',node_features.shape)
        # print('edge_features',edge_features.shape)
        
        node_out = self.propagate(edge_index, x=(node_feat_i, node_feat_j), size=size, edge_features=edge_feat_ij)  
        node_out = self.node_mlp(node_out.view(-1, H * C))
        
        edge_out = self.edge_transform(edge_features) + self.edge_mlp(self.edge_out.view(-1, H * C))
        # node_out += self.node_transform(node_features)
        node_out = node_out + self.node_transform(node_features)
        
        if self.return_attn_weights:
            return node_out, edge_out, self.attn_weights
        else:
            return node_out, edge_out

    def message(self, x_i, x_j, index, ptr, size_i, edge_features):
        f_ij = torch.cat([x_i, edge_features, x_j], dim=-1)  # shape [E, H*C]
        f_ij = self.attn_mlp(f_ij)
        f_ij = F.leaky_relu(f_ij, negative_slope=self.negative_slope).view(-1, self.heads, self.out_features)
        
        self.edge_out = f_ij  # Multi-head edge features
        
        if self.use_attention_weights:
            attention_scores = (f_ij * self.attn_param).sum(dim=-1).unsqueeze(-1)
        else:
            attention_scores = f_ij.sum(dim=-1).unsqueeze(-1)
            
        alpha = pyg_utils.softmax(attention_scores, index, ptr, size_i)  # Normalized attention weights
        alpha = F.dropout(alpha, p=self.dropout)
        
        # self.attn_weights = alpha  # shape [E, H, 1]
        self.attn_weights = self.linear(alpha.squeeze(-1)).view(-1)  # shape [E]
        
        return x_j.view(-1, self.heads, self.out_features) * alpha

    def aggregate(self, inputs, index, dim_size=None):
        return scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)



class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, edge_norm=True, gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        loop_weight = torch.full((num_nodes,), 1 if not improved else 2, dtype=edge_weight.dtype, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        if torch.any(deg < 0):
            print("There are negative values in deg.")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x
        # if torch.isnan(x).any():
            # print('before x has NaN')

        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        # if torch.isnan(norm).any():
            # print('after norm has NaN')
        out = self.propagate(edge_index, x=x, norm=norm)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
    

class EGATGCNConv(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_features, heads=3, layers=3, edge_norm=True, gfn=False):
        super(EGATGCNConv, self).__init__()
        self.egat = EGAT(in_node_features, in_edge_features, out_features, heads, layers)
        self.gcn = GCNConv(out_features, out_features, edge_norm=edge_norm, gfn=gfn)

    def forward(self, node_features, edge_index, edge_features, edge_weight=None):
        
        node_out, edge_out, attn_weights = self.egat(node_features, edge_index, edge_features)

        
        out = self.gcn(node_out, edge_index, edge_weight=edge_weight)
        # if torch.isnan(out).any():
            # print('node_out has NaN')
        return out, edge_out, attn_weights
