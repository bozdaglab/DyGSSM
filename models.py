from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from config import cfg
from torch_scatter import scatter_add
from torch.autograd import Variable

"""GCN block"""
class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out, pos_isn):
        super(GCNLayer, self).__init__()
        self.pos_isn = pos_isn
        if cfg.gnn.skip_connection == 'affine':
            self.linear_skip_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
            self.linear_skip_bias = nn.Parameter(torch.ones(size=(dim_out, )))
        elif cfg.gnn.skip_connection == 'identity':
            assert self.dim_in == self.out_channels

        self.linear_msg_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
        self.linear_msg_bias = nn.Parameter(torch.ones(size=(dim_out, )))

        self.activate = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight, gain=gain)
        nn.init.xavier_normal_(self.linear_msg_weight, gain=gain)

        nn.init.constant_(self.linear_skip_bias, 0)
        nn.init.constant_(self.linear_msg_bias, 0)

    def norm(self, graph):
        edge_index = graph.edges()
        row = edge_index[0]
        edge_weight = torch.ones((row.size(0),),
                                 device=row.device)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=graph.num_nodes())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt

    def message_fun(self, edges):
        return {'m': edges.src['h'] * 0.1}

    def forward(self, g, feats, fast_weights=None):
        if fast_weights:
            linear_skip_weight = fast_weights[0]
            linear_skip_bias = fast_weights[1]
            linear_msg_weight = fast_weights[2]
            linear_msg_bias = fast_weights[3]

        else:
            linear_skip_weight = self.linear_skip_weight
            linear_skip_bias = self.linear_skip_bias
            linear_msg_weight = self.linear_msg_weight
            linear_msg_bias = self.linear_msg_bias



        feat_src, feat_dst = expand_as_pair(feats, g)
        norm_ = self.norm(g)
        feat_src = feat_src * norm_.view(-1, 1)
        g.srcdata['h'] = feat_src
        aggregate_fn = fn.copy_u('h', 'm')

        g.update_all(message_func=aggregate_fn, reduce_func=fn.sum(msg='m', out='h'))
        rst = g.dstdata['h']
        rst = rst * norm_.view(-1, 1)

        rst_ = F.linear(rst, linear_msg_weight, linear_msg_bias)

        if cfg.gnn.skip_connection == 'affine':
            skip_x = F.linear(feats, linear_skip_weight, linear_skip_bias)

        elif cfg.gnn.skip_connection == 'identity':
            skip_x = feats

        return rst_ + skip_x


class Model(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, num_layers, dropout, CONFIG):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        for i in range(num_layers):
            d_in = in_features if i == 0 else out_features
            pos_isn = True if i == 0 else False
            layer = GCNLayer(d_in, out_features, pos_isn)
            self.add_module('layer{}'.format(i), layer)

        self.weight1 = nn.Parameter(torch.ones(size=(hidden_dim, out_features)))
        self.weight2 = nn.Parameter(torch.ones(size=(1, hidden_dim)))

        if cfg.model.edge_decoding == 'dot':
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        elif cfg.model.edge_decoding == 'cosine_similarity':
            self.decode_module = nn.CosineSimilarity(dim=-1)
        else:
            raise ValueError('Unknown edge decoding {}.'.format(
                cfg.model.edge_decoding))
        self.reset_parameters()
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, g, x, fast_weights, seq_embeddings, cross_attention):
        count = 0
        for layer in self.children():
            if not isinstance(layer, CrossAttention):
                x = layer(g, x, fast_weights[2 + count * 4: 2 + (count + 1) * 4])
                count += 1

        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2
        # x = x + seq_embeddings
        x, weights_1, weights_2 = cross_attention(x, seq_embeddings, seq_embeddings)
        x = F.normalize(x)
        g.node_embedding = x

        # pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(x, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = F.sigmoid(F.linear(pred, weight2))

        node_feat = pred[g.edge_label_index]
        nodes_first = node_feat[0]
        nodes_second = node_feat[1]

        pred = self.decode_module(nodes_first, nodes_second)

        return pred, g.node_embedding
    


"""GRU block"""
class SeqRW(nn.Module):
    def __init__(
        self, inp_size, emb_dim, hidden, num_layers, dropout) -> None:
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = False
        self.emb = nn.Embedding(num_embeddings=inp_size, embedding_dim=emb_dim)
        self.seq = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            batch_first=False,
        )
        if cfg.model.edge_decoding == "dot":
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        elif cfg.model.edge_decoding == "cosine_similarity":
            self.decode_module = nn.CosineSimilarity(dim=-1)
        if self.bidirectional:
            self.linear = nn.Linear(in_features=hidden * 2, out_features=1)
        else:
            self.linear = nn.Linear(in_features=hidden, out_features=1)

    def forward(self, graph, device):
        walk_seq = graph.random_walk_node2vec_f
        seq_embeddings = self.emb(walk_seq.to(device))
        output, hidden_satet = self.seq(seq_embeddings)
        last_out = output[:, -1, :]
        last_out1 = F.dropout(last_out, self.dropout)
        last_out1 = F.sigmoid(self.linear(last_out))
        node_feat = last_out1[graph.edge_label_index]
        nodes_first = node_feat[0]
        nodes_second = node_feat[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, last_out


class GRUCellSimpler(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellSimpler, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.linear_skip_weight_gru_ir = nn.Parameter(torch.ones(size=(hidden_size, hidden_size)))
        self.linear_skip_bias_gru_ir = nn.Parameter(torch.ones(size=(hidden_size, )))
        self.linear_skip_weight_gru_iz = nn.Parameter(torch.ones(size=(hidden_size, hidden_size)))
        self.linear_skip_bias_gru_iz = nn.Parameter(torch.ones(size=(hidden_size, )))
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight_gru_ir, gain=gain)
        nn.init.xavier_normal_(self.linear_skip_weight_gru_iz, gain=gain)


    def forward(self, x, hidden):
        linear_skip_weight_gru_ir = self.linear_skip_weight_gru_ir
        linear_skip_bias_gru_ir = self.linear_skip_bias_gru_ir
        linear_skip_weight_gru_iz = self.linear_skip_weight_gru_iz
        linear_skip_bias_gru_iz = self.linear_skip_bias_gru_iz


        h_t = F.linear(x, linear_skip_weight_gru_ir, linear_skip_bias_gru_ir)
        z_t = F.sigmoid(F.linear(x, linear_skip_weight_gru_iz, linear_skip_bias_gru_iz))

        return  (1 - z_t) * hidden + z_t * h_t


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        for i in range(num_layer):
            d_in = input_dim if i == 0 else hidden_dim
            d_ou = hidden_dim if i == 0 else output_dim
            layer = GRUCellSimpler(d_in, d_ou)
            self.add_module(f'layer_sequence{i}', layer)
        self.dropout = 0.2

        
        self.weight_gru_skip = nn.Parameter(torch.rand(size=(output_dim, hidden_dim)))
        self.weight_gru_bias = nn.Parameter(torch.rand(size=(output_dim, )))
        self.emb = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim)
        if cfg.model.edge_decoding == 'dot':
            self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        elif cfg.model.edge_decoding == 'cosine_similarity':
            self.decode_module = nn.CosineSimilarity(dim=-1)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, graph):
        x = self.emb(graph.random_walk_node2vec_f.to("cuda:2"))
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim))
        hn = h0
        count = 0
        for layer in self.children():
            if isinstance(layer, GRUCellSimpler):
                for seq in range(x.size(1)):
                    x1 = x[:,seq,:]
                    hn = layer(x=x1, 
                            hidden=hn
                            )
            count += 1
        last_out1 = F.dropout(hn, self.dropout)
        last_out1 = F.sigmoid(self.linear(hn))
        node_feat = last_out1[graph.edge_label_index]
        nodes_first = node_feat[0]
        nodes_second = node_feat[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, F.linear(hn, self.weight_gru_skip, self.weight_gru_bias)



"""Crross attention block"""
class CrossAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, num_cross_att
    ) -> None:
        super().__init__()
        self.num_cross_att = num_cross_att
        self.multiattention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.parameter = nn.Parameter(torch.rand(1))

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        out_attention_1, weights_1 = self.multiattention(query, key, value)
        cross_attention_1 = self.norm_1(self.dropout(out_attention_1) + query)
        if self.num_cross_att == 1:
            return cross_attention_1, weights_1, "weights_2"
        out_attention_2, weights_2 = self.multiattention(key, query, query)
        cross_attention_2 = self.norm_1(self.dropout(out_attention_2) + key)
        combined = (
            self.parameter * cross_attention_1
            + (1 - self.parameter) * cross_attention_2
        )
        return combined, weights_1, weights_2


class CrossAttentionCuston(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_cross_attention: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.num_cross_attention = num_cross_attention
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(embed_dim)
        self.parameter = nn.Parameter(torch.rand(1))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query_1 = self.W_q(query)
        key_1 = self.W_k(key)
        value_1 = self.W_v(value)

        query_head_1 = query_1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_head_1 = value_1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_head_1 = key_1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores_1 = torch.matmul(query_head_1, key_head_1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores_1.masked_fill_(mask == 0, float('-inf'))

        attention_weights_1 = F.softmax(scores_1, dim=-1)
        attention_output_1 = torch.matmul(attention_weights_1, value_head_1)
        attention_output_1 = attention_output_1.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        cross_att_1 = self.W_o(self.norm(self.dropout(attention_output_1.squeeze(1)) + query))
        if self.num_cross_attention == 1:
            return cross_att_1, attention_weights_1, "weights"




"""SSM Block"""    
class HIPPO(torch.nn.Module):
    def __init__(self, state_dim):
        """ Initialize HIPPO state for each independent component. """
        super().__init__()
        self.state_dim = state_dim
        self.state = torch.zeros(state_dim)
        self.A = self.create_projection_matrix(state_dim)

    def create_projection_matrix(self, n):
        """ Create projection matrix based on Legendre polynomials. """
        A = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if i> j:
                    A[i, j] = (-1)**(i-j) * (2*i+1)
                elif i ==j:
                    A[i, j] = 2
                else:
                    A[i, j] = 0 #(2 * i + 1) ** 0.5
        # A = A / torch.linalg.norm(A)
        return A

    def forward(self, grad_vector, loss):
        """ Update HIPPO state with gradient vector. """
        try:
            self.state = torch.matmul(self.A, self.state) + grad_vector * loss#[:self.state_dim]
        except:
            self.state = torch.matmul(self.A[0], self.state) + grad_vector * loss#[:self.state_dim]
        return self.state

