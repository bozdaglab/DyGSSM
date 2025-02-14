from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair
from config import cfg
import math
from collections import defaultdict
from torch_scatter import scatter_add
from einops import rearrange, repeat


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


class HiPPOWithGradients(nn.Module):
    def __init__(self, param_shapes, method="Legendre", lr=0.01):
        """
        Combine HiPPO memory dynamics with gradient updates.
        
        Args:
            param_shapes (list of torch.Size): List of parameter shapes.
            method (str): HiPPO method ("Legendre", etc.).
            lr (float): Learning rate for gradient descent.
        """
        super(HiPPOWithGradients, self).__init__()
        self.param_shapes = param_shapes
        self.lr = lr  # Learning rate for gradient updates
        
        # Initialize HiPPO states and projection matrices for each parameter
        self.hippo_states = nn.ParameterList()
        self.projection_matrices = nn.ParameterList()
        
        for shape in param_shapes:
            # Flatten each parameter to a 1D vector
            state_dim = torch.prod(torch.tensor(shape)).item()
            
            # HiPPO memory state
            self.hippo_states.append(nn.Parameter(torch.zeros(state_dim), requires_grad=False))
            
            # HiPPO projection matrix (A)
            A = self._initialize_projection_matrix(state_dim, method)
            self.projection_matrices.append(nn.Parameter(A, requires_grad=False))
    
    def _initialize_projection_matrix(self, state_dim, method="Legendre"):
        """
        Efficient initialization of HiPPO projection matrix.
        """
        # Use sparse representation for Legendre
        if method == "Legendre":
            indices = torch.arange(0, state_dim, dtype=torch.float32) * 2 + 1
            return torch.diag(indices)
        else:
            raise NotImplementedError(f"HiPPO method '{method}' not implemented.")
    def forward(self, parameters, gradients, weights):
        """
        Update parameters using HiPPO memory and gradients.
        
        Args:
            parameters (list of Tensors): Current parameters.
            gradients (list of Tensors): Gradients for the parameters.
        
        Returns:
            list of Tensors: Updated parameters.
        """
        updated_parameters = []
        for param, grad, state, A in zip(parameters, gradients, self.hippo_states, self.projection_matrices):
            # Flatten parameter and gradient
            param_flat = param.flatten()
            grad_flat = grad.flatten()
            
            # HiPPO update: h_t = A @ h_t-1 + param
            state[:] = torch.matmul(A, state) + param_flat
            
            # Gradient descent: theta = h_t - lr * grad
            updated_param_flat = state - weights * grad_flat
            
            # Reshape back to original parameter shape
            updated_param = updated_param_flat.view(param.shape)
            updated_parameters.append(updated_param)
        
        return updated_parameters
    
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


class HiPPODiscretizationSSM(nn.Module):
    def __init__(self, 
                 state_dims):
        super().__init__()

        self.activation = "silu"
        self.act = nn.SiLU()
        self.mamba_states = defaultdict(dict)

        for shape_idx, shape in enumerate(state_dims):
            if len(shape) > 1:
              d_model = shape[1]
              self.state = torch.ones(shape[0], shape[1]).to("cuda")
              self.norm = nn.LayerNorm(shape[1]).to("cuda")
            else:
                d_model = shape[0]
                self.state = torch.ones(shape[0]).to("cuda")
                self.norm = nn.LayerNorm(shape[0]).to("cuda")
            self.param1 = nn.Linear(d_model * 2, d_model).to("cuda")
            self.mamba_states[shape_idx]["self.param1"] = self.param1
            self.param2 = nn.Linear(d_model * 2, d_model).to("cuda")
            self.mamba_states[shape_idx]["self.param2"] = self.param2
            self.param3 = nn.Linear(d_model * 2, d_model).to("cuda")
            self.mamba_states[shape_idx]["self.param3"] = self.param3
            self.mamba_states[shape_idx]["self.state"] = self.state
            self.mamba_states[shape_idx]["self.norm"] = self.norm

    def forward(self, hidden_states, adjs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        next_state = []
        for idx, state in enumerate(hidden_states):
            state_new = torch.cat((state, adjs[idx]), dim=-1)
            i =  self.act(self.mamba_states[idx]["self.param1"](state_new))
            f =  self.act(self.mamba_states[idx]["self.param2"](state_new))

            c_bar = self.act(self.mamba_states[idx]["self.param3"](state_new))
            c = self.mamba_states[idx]["self.norm"](f * self.mamba_states[idx]["self.state"] + i * c_bar)
            self.mamba_states[idx]["self.state"] = c
            next_state.append(c)
        return next_state




class HiPPODiscretizationRNN(nn.Module):
    def __init__(self, 
                 state_dims):
        super().__init__()

        self.activation = "silu"
        self.act = nn.SiLU()
        self.mamba_states = defaultdict(dict)

        for shape_idx, shape in enumerate(state_dims):
            if len(shape) > 1:
              d_model = shape[1]
            #   self.state = torch.ones(shape[0], shape[1]).to("cuda")
              self.norm = nn.LayerNorm(shape[1]).to("cuda")
            else:
                d_model = shape[0]
                # self.state = torch.ones(shape[0]).to("cuda")
                self.norm = nn.LayerNorm(shape[0]).to("cuda")
            self.param1 = nn.Linear(d_model * 2, d_model).to("cuda")
            self.mamba_states[shape_idx]["self.param1"] = self.param1
            self.param2 = nn.Linear(d_model * 2, d_model).to("cuda")
            self.mamba_states[shape_idx]["self.param2"] = self.param2
            self.param3 = nn.Linear(d_model * 2, d_model).to("cuda")
            self.mamba_states[shape_idx]["self.param3"] = self.param3
            # self.mamba_states[shape_idx]["self.state"] = self.state
            self.mamba_states[shape_idx]["self.norm"] = self.norm

    def forward(self, hidden_states, adjs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        next_state = []
        for idx, state in enumerate(hidden_states):
            state_new = torch.cat((adjs[idx], state), dim=-1)
            z =  self.act(self.mamba_states[idx]["self.param1"](state_new))
            r =  self.act(self.mamba_states[idx]["self.param2"](state_new))
            h_prime_t = self.act(self.mamba_states[idx]["self.param3"](torch.cat((adjs[idx],  (r * state)), dim=-1)))
            h_t= z * state + (1 - z) * h_prime_t
            # c_bar = self.act(self.mamba_states[idx]["self.param3"](state_new))
            # c = self.mamba_states[idx]["self.norm"](f * self.mamba_states[idx]["self.state"] + i * c_bar)
            # self.mamba_states[idx]["self.state"] = c
            next_state.append(h_t)
        return next_state
    

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices for update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # Weight matrices for reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        # Weight matrices for candidate hidden state
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=-1)
        
        # Compute gates and candidate hidden state
        z = torch.sigmoid(self.W_z(combined))  # Update gate
        r = torch.sigmoid(self.W_r(combined))  # Reset gate
        
        combined_reset = torch.cat([x, r * h_prev], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_reset))

        # Compute new hidden state
        h_next = (1 - z) * h_prev + z * h_candidate
        return h_next













class HiPPODiscretization_3(nn.Module):
    def __init__(self, 
                 state_dims,  
                 dt_max, 
                 dt_min, 
                 dt_init_floor, 
                 device, 
                 d_conv=4,
                 dt_scale=1.0, 
                 dt_rank="auto", 
                 dt_init="random",
                 conv_bias=True,
                 bias=False, 
                 dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mamba_states = defaultdict(dict)
        self.d_state = 16
        self.activation = "silu"
        self.act = nn.SiLU()

        for shape_idx, shape in enumerate(state_dims):
            self.d_model = shape[1] if len(shape)>1 else shape[0]
            self.mamba_states[shape_idx]["self.d_model"] = self.d_model 
            self.d_inner = shape[1] if len(shape)>1 else shape[0]
            self.mamba_states[shape_idx]["self.d_inner"] = self.d_inner 
            self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
            self.mamba_states[shape_idx]["self.dt_rank"] = self.dt_rank
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.mamba_states[shape_idx]["self.in_proj"] = self.in_proj
            self.x_proj = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.mamba_states[shape_idx]["self.x_proj"] = self.x_proj
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.mamba_states[shape_idx]["self.dt_proj"] = self.dt_proj
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.mamba_states[shape_idx]["self.conv1d"] = self.conv1d
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.mamba_states[shape_idx]["self.dt_proj"] = self.dt_proj
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_proj.bias._no_reinit = True
            self.mamba_states[shape_idx]["self.dt_proj"] = self.dt_proj
            # S4D real initialization
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            self.mamba_states[shape_idx]["self.A_log"] = self.A_log

            # D "skip" parameter
            self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D._no_weight_decay = True
            self.mamba_states[shape_idx]["self.D"] = self.D
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
            self.mamba_states[shape_idx]["self.out_proj"] = self.out_proj
            
    def forward(self, hidden_states, adjs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        next_state = []
        for idx, state in enumerate(hidden_states):
            state_new = torch.stack((state, adjs[idx]))
            if len(state_new.shape) != 3:
                 state_new = state_new.unsqueeze(1)

            batch, seqlen, dim = state_new.shape

            conv_state, ssm_state = None, None
            xz = rearrange(
                self.mamba_states[idx]["self.in_proj"].weight @ rearrange(state_new, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

            A = -torch.exp(self.mamba_states[idx]["self.A_log"].float())  # (d_inner, d_state)
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.mamba_states[idx]["self.conv1d"](x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.mamba_states[idx]["self.conv1d"].weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            x_dbl = self.mamba_states[idx]["self.x_proj"](rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, 
                                   [self.mamba_states[idx]["self.dt_rank"], 
                                    self.d_state, 
                                    self.d_state], dim=-1)
            dt = self.mamba_states[idx]["self.dt_proj"].weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)


            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.mamba_states[idx]["self.D"].float(),
                z=z,
                delta_bias=self.mamba_states[idx]["self.dt_proj"].bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            y = y[1:, :, :].squeeze(0)
            out = self.mamba_states[idx]["self.out_proj"](y)
            next_state.append(out)
        return next_state



class HiPPODiscretization_2(nn.Module):
    def __init__(self, state_dims, dt_min, dt_max, dt_init_floor, param_c=False):
        super().__init__()
        self.state_dims = state_dims
        self.param_c = param_c

        self.A_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
            self.A_list.append(param)

        self.B_list = []
        self.layer_norm_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
                self.layer_norm_list.append(nn.LayerNorm(state_dim[1]).to("cuda"))
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
                self.layer_norm_list.append(nn.LayerNorm(state_dim[0]).to("cuda"))
            self.B_list.append(param)
            


        self.C_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
            self.C_list.append(param)


        self.delta_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
            self.delta_list.append(param)

        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt = nn.ParameterList()
        for state_dim in state_dims:
            try:
                dt_val = torch.exp(torch.rand(state_dim[1]) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=self.dt_init_floor)
            except:
                dt_val = torch.exp(torch.rand(state_dim[0]) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=self.dt_init_floor)
            self.dt.append(dt_val)

    def forward(self, prev_state_list, current_input_list):
        next_state_list = []
        for i, prev_state in enumerate(prev_state_list):
            B = self.B_list[i](prev_state)
            # C = self.C_list[i](prev_state)
            # A = self.A_list[i](prev_state)
            delta = torch.log(1 + torch.exp(self.delta_list[i](prev_state)) + self.dt[i])
            delta_A = torch.multiply(delta, B)
            delta_B = torch.multiply(delta, B)
            hid_state = torch.multiply(delta_A, prev_state) + torch.multiply(delta_B , current_input_list[i])
            # if self.param_c:
            #     hid_state = torch.multiply(hid_state , C)
            # hid_state = hid_state + prev_state
            hid_state = self.layer_norm_list[i](hid_state)
            next_state_list.append(hid_state)
        return next_state_list
    




class HiPPODiscretization_1(nn.Module):
    def __init__(self, state_dims, dt_min, dt_max, dt_init_floor, param_c=False):
        super().__init__()
        self.state_dims = state_dims
        self.param_c = param_c
        # self.A_list = []
        # for state_dim in state_dims:
        #     try:
        #         param = nn.Parameter(torch.randn(state_dim[0], state_dim[1])).to("cuda")
        #     except:
        #         param = nn.Parameter(torch.randn(state_dim[0], state_dim[0])).to("cuda")
        #     self.A_list.append(param)
        list_shape = []
        for shape in state_dims:
            s1 = shape[0]
            try:
                s2 = shape[1]
                list_shape.append(0 if s1>s2 else 1)
            except IndexError:
                list_shape.append(0)
        self.A_list = [nn.Parameter(torch.randn(state_dim[list_shape[idx]], state_dim[list_shape[idx]])).to("cuda") for idx, state_dim in enumerate(state_dims)]
        self.B_list = []
        self.layer_norm_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
                self.layer_norm_list.append(nn.LayerNorm(state_dim[1]).to("cuda"))
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
                self.layer_norm_list.append(nn.LayerNorm(state_dim[0]).to("cuda"))
            self.B_list.append(param)
            


        self.C_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
            self.C_list.append(param)


        self.delta_list = []
        for state_dim in state_dims:
            try:
                param = nn.Linear(state_dim[1], state_dim[1]).to("cuda")
            except:
                param = nn.Linear(state_dim[0], state_dim[0]).to("cuda")
            self.delta_list.append(param)

        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt = nn.ParameterList()
        for state_dim in state_dims:
            try:
                dt_val = torch.exp(torch.rand(state_dim[1]) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=self.dt_init_floor)
            except:
                dt_val = torch.exp(torch.rand(state_dim[0]) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)).clamp(min=self.dt_init_floor)
            self.dt.append(dt_val)

    def forward(self, prev_state_list, current_input_list):
        next_state_list = []
        for i, prev_state in enumerate(prev_state_list):
            B = self.B_list[i](prev_state)
            C = self.C_list[i](prev_state)
            delta = torch.log(1 + torch.exp(self.delta_list[i](prev_state)) + self.dt[i])
            try:
                delta_A = self.A_list[i] @ delta
            except RuntimeError:
                delta_A = delta @ self.A_list[i]
            delta_B = torch.multiply(delta, B)
            hid_state = torch.multiply(delta_A, prev_state) + torch.multiply(delta_B , current_input_list[i])
            if self.param_c:
                hid_state = torch.multiply(hid_state , C)
            # hid_state = hid_state + prev_state
            hid_state = self.layer_norm_list[i](hid_state)
            next_state_list.append(hid_state)
        return next_state_list
    





"""Multi Loss Aggregation"""
class MultiLossAggregation(nn.Module):
    def __init__(self, win_size):
        super().__init__()
        self.sigma = nn.Parameter(torch.ones(win_size))
    
    def forward(self, losses):
        # loss = 0.5 / (self.sigma**2) * losses
        # loss = loss.sum() + torch.log(self.sigma.prod())
        loss_sum = 0
        for i, loss in enumerate(losses):
            # Compute the weighted loss component for each task
            weighted_loss = 0.5 / (self.sigma[i] ** 2) * loss
            # Add a regularization term to encourage the learning of useful weights
            regularization = torch.log(1 + self.sigma[i] ** 2)
            # Sum the weighted loss and the regularization term
            loss_sum += weighted_loss + regularization
        return loss
