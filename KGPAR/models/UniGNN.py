import torch
import torch.nn as nn, torch.nn.functional as F


import math 

from torch_scatter import scatter
from torch_geometric.utils import softmax


# NOTE: can not tell which implementation is better statistically 

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X



# v1: X -> XW -> AXW -> norm
class UniSAGEConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce=self.args.second_aggregate, dim_size=N) # [N, C]
        X = X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.]))
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


    def forward(self, X, vertex, edges):
        N = X.shape[0]
        # X0 = X # NOTE: reserved for skip connection
        
        # v1: X -> XW -> AXW -> norm
        X = self.W(X) 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X = (1 + self.eps) * X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        
        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGCNConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV
        
        # v1: X -> XW -> AXW -> norm
        
        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 
        
        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: skip concat here?

        return X



# v2: X -> AX -> norm -> AXW 
class UniGCNConv2(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v3: X -> AX -> norm -> AXW 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        X = self.W(X)


        # NOTE: result might be slighly unstable
        # NOTE: skip concat here?

        return X



class UniGATConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex] # [nnz, H, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, H, C]


        alpha_e = (Xe * self.att_e).sum(-1) # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop( alpha )
        alpha = alpha.unsqueeze(-1)


        Xev = Xe[edges] # [nnz, H, C]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, H, C]
        X = Xv 
        X = X.view(N, H * C)

        if self.args.use_norm:
            X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0 

        # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return X




__all_convs__ = {
    'UniGAT': UniGATConv,
    'UniGCN': UniGCNConv,
    'UniGCN2': UniGCNConv2,
    'UniGIN': UniGINConv,
    'UniSAGE': UniSAGEConv,
}



class UniGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )
        self.V = V 
        self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        V, E = self.V, self.E 
        
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)
    
        return X



class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    
    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV
        
        X = Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)


        return X



class UniGCNII(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V 
        self.E = E 
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        V, E = self.V, self.E 
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return x


