import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn



class DyGSSM(nn.Module):
    def __init__(self,
                n_noeds, 
                in_features,
                out_features, 
                hidden_dim, 
                hidden, 
                num_layers, 
                dropout,
                num_heads, 
                device,
                fused_model,
                train_type,
                bidirectional,
                message_pass_type):
        super().__init__()
        self.bidirectional = bidirectional

        if message_pass_type == "GCN":
            self.msp1 = dglnn.GraphConv(in_features, out_features, allow_zero_in_degree=True) 
            self.msp2 = dglnn.GraphConv(out_features, out_features, allow_zero_in_degree=True)
        elif message_pass_type == "GraphSage":
            self.msp1 = dglnn.SAGEConv(in_features, out_features, aggregator_type="pool")
            self.msp2 = dglnn.SAGEConv(out_features, out_features, aggregator_type="pool")
        elif message_pass_type == "GAT":
            self.msp1 = dglnn.GATConv(in_features, out_features, num_heads=num_heads, allow_zero_in_degree=True)
            self.msp2 = dglnn.GATConv(out_features, out_features, num_heads=num_heads, allow_zero_in_degree=True)
        self.train_type = train_type
        self.mlp = nn.Sequential(
            nn.Linear(in_features=out_features * 2, out_features=int(out_features)),
            self.create_activation("relu"),
            nn.Dropout(dropout),
            nn.Linear(in_features=int(out_features), out_features=int(out_features / 2)),
            self.create_activation("relu"),
            nn.Dropout(dropout),
            nn.Linear(in_features=int(out_features / 2), out_features=1),
        )
        self.norm = nn.LayerNorm(out_features)
        self.activation = self.create_activation("relu")

        self.init_embeddings = nn.Linear(in_features, out_features)
        self.emb = nn.Linear(out_features, hidden_dim)
        self.seq = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1)
        self.reverse_emb = nn.Linear(hidden_dim, out_features)
        self.cross_attention = SemanticAttention(hidden, dropout)

        self.device = device

        self.initialize_parameters()
    def initialize_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if isinstance(self.msp1, dglnn.GraphConv):
            nn.init.xavier_uniform_(self.msp1.weight, gain=gain)
            nn.init.xavier_uniform_(self.msp2.weight, gain=gain)
            
        elif isinstance(self.msp1, dglnn.SAGEConv):
            if hasattr(self.msp1, "fc_pool"):
                nn.init.xavier_uniform_(self.msp1.fc_pool.weight, gain=gain)
                if self.msp1.fc_pool.bias is not None:
                    nn.init.zeros_(self.msp1.fc_pool.bias)
                nn.init.xavier_uniform_(self.msp2.fc_pool.weight, gain=gain)
                if self.msp2.fc_pool.bias is not None:
                    nn.init.zeros_(self.msp2.fc_pool.bias)
            if hasattr(self.msp1, "fc_self"):
                nn.init.xavier_uniform_(self.msp1.fc_self.weight, gain=gain)
                if self.msp1.fc_self.bias is not None:
                    nn.init.zeros_(self.msp1.fc_self.bias)
                nn.init.xavier_uniform_(self.msp2.fc_pool.weight, gain=gain)
                if self.msp2.fc_self.bias is not None:
                    nn.init.zeros_(self.msp2.fc_pool.bias)
            if hasattr(self.msp1, "fc_neigh"):
                nn.init.xavier_uniform_(self.msp1.fc_neigh.weight, gain=gain)
                if self.msp1.fc_neigh.bias is not None:
                    nn.init.zeros_(self.msp1.fc_neigh.bias)
                nn.init.xavier_uniform_(self.msp2.fc_pool.weight, gain=gain)
                if self.msp2.fc_neigh.bias is not None:
                    nn.init.zeros_(self.msp2.fc_pool.bias)

        elif isinstance(self.msp1, dglnn.GATConv):
            gain = nn.init.calculate_gain("leaky_relu")
            if hasattr(self.msp1, "fc"):
                nn.init.xavier_uniform_(self.msp1.fc.weight, gain=gain)
                if self.msp1.fc.bias is not None:
                    nn.init.zeros_(self.msp1.fc.bias)
                nn.init.xavier_uniform_(self.msp2.fc.weight, gain=gain)
                if self.msp2.fc.bias is not None:
                    nn.init.zeros_(self.msp2.fc.bias)
            if hasattr(self.msp1, "attn_l"):
                nn.init.xavier_uniform_(self.msp1.attn_l, gain=gain)
                nn.init.xavier_uniform_(self.msp2.attn_l, gain=gain)
            if hasattr(self.msp1, "attn_r"):
                nn.init.xavier_uniform_(self.msp1.attn_r, gain=gain)
                nn.init.xavier_uniform_(self.msp2.attn_r, gain=gain)
                
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
        
        nn.init.kaiming_normal_(self.emb.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb.bias)
        nn.init.kaiming_normal_(self.reverse_emb.weight, nonlinearity='relu')
        nn.init.zeros_(self.reverse_emb.bias)

        nn.init.kaiming_normal_(self.init_embeddings.weight, nonlinearity='relu')
        nn.init.zeros_(self.init_embeddings.bias)
        
        nn.init.kaiming_normal_(self.seq.weight, nonlinearity='relu')
        nn.init.zeros_(self.seq.bias)

    def create_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "softmax":
            return nn.Softmax()
        raise RuntimeError("activation error, not {}".format(activation))

    def forward(self, graph, x):
        x_out = self.msp1(graph, x)
        if isinstance(self.msp1, dglnn.GATConv):
            x_out = x_out.mean(dim=1)
        x_out = self.activation(self.norm(x_out))
        sec_out = self.msp2(graph, x_out)
        if isinstance(self.msp1, dglnn.GATConv):
            sec_out = sec_out.mean(dim=1)
        x_out = self.norm(sec_out + x_out)   

        walk_seq = graph.random_walk_node2vec_f
        init_emb = self.init_embeddings(x.to("cuda:0"))
        seq_embeddings = self.emb(init_emb[walk_seq])
        out_rw  = self.seq(seq_embeddings.to(self.device)).squeeze(1)
        last_out = self.reverse_emb(out_rw)
        inp = torch.stack([x_out, last_out])
        x_fused, _ = self.cross_attention(inp)
        x_fused = x_fused.mean(dim=1)
        if self.train_type == "wingnn":
            node_feat = x_fused[graph.edge_label_index]
            nodes_first = node_feat[0]
            nodes_second = node_feat[1]
            torch.cuda.empty_cache()
            pred = self.mlp(torch.concat((nodes_first, nodes_second), dim=-1))
            return pred.squeeze(-1)
        elif self.train_type == "hawkes":
            return x_fused
        else:
            raise ValueError

    def predict(self, h, edge_index, bs=1024*64):
        out = []
        for edge in torch.split(edge_index, bs, dim=1):
            out.append(self.mlp(torch.concat((h[edge[0]], h[edge[1]]), dim=-1)).squeeze(-1))
        return torch.cat(out)

class SemanticAttention(nn.Module):
    def __init__(self, in_dim, dropout, dim_a=50, num_relations=2):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.num_relations)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h, return_attn=False):
        attention = F.softmax(
            torch.matmul(
                torch.sigmoid(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).permute(1, 0, 2)

        attention = self.dropout(attention)

        # Output shape: (batch_size, num_relations, dim)
        h = torch.matmul(attention, h.permute(1, 0, 2))

        return h, attention if return_attn else None




# class HIPPO(nn.Module):
#     def __init__(self, grad_dim, state_dim):
#         super().__init__()
#         self.grad_dim = grad_dim
#         self.state_dim = state_dim
#         self.A = self.create_hippo_legs_matrix(state_dim)
#         self.B = torch.sqrt(torch.tensor(2 * torch.arange(state_dim) + 1.0)).unsqueeze(1)
#         self.output_proj = nn.Linear(state_dim, grad_dim)
#         self.trigger_head = nn.Sequential(
#             nn.Linear(state_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#         self.state = torch.zeros(state_dim)

#     def create_hippo_legs_matrix(self, N):
#         A = torch.zeros(N, N)
#         for n in range(N):
#             for m in range(N):
#                 if n >= m:
#                     A[n, m] = (2 * n + 1) * (1 if n == m else (-1) ** (n - m))
#         A *= -1
#         return nn.Parameter(A, requires_grad=True)

#     def forward(self, grad_vector, weight):
#         u = grad_vector * weight  # scalar
#         Bu = self.B.squeeze() * u
#         self.state = torch.matmul(self.state, self.A.T) + Bu  # [state_dim]
#         modulation = self.output_proj(self.state)             # [grad_dim]
#         trigger = self.trigger_head(self.state)               # [1] ∈ (0, 1)
#         return modulation, trigger
    
"""SSM Block"""    
class HIPPO(torch.nn.Module):
    def __init__(self, state_dim):
        """ Initialize HIPPO state for each independent component. """
        super().__init__()
        self.state_dim = state_dim
        self.state = torch.zeros(state_dim)
        self.A = self.create_projection_matrix(state_dim)
        # self.gate_net = nn.Sequential(
        #     nn.Linear(state_dim, state_dim),
        #     nn.Sigmoid()
        # ).cuda()

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
            self.state = torch.matmul(self.A.cuda(), self.state) + grad_vector * loss#[:self.state_dim]
        except:
            self.state = torch.matmul(self.A[0], self.state) + grad_vector * loss#[:self.state_dim]
        return self.state
