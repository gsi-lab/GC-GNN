import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Set2Set(nn.Module):
    def __init__(self, in_dim, device, num_iters=6, num_layers=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 2 * in_dim
        self.num_layers = num_layers
        self.num_iters = num_iters
        self.device = device
        self.lstm_output_dim = self.out_dim - self.in_dim
        self.lstm = nn.LSTM(self.out_dim, self.in_dim, num_layers=num_layers, batch_first=True)
        #self.predict = nn.Sequential(
        #    nn.Linear(self.out_dim, self.in_dim),
        #    nn.ReLU()
        #)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        #self.predict[0].reset_parameters()

    def forward(self, graph, feats):
        with graph.local_scope():
            batch_size = graph.batch_size
            hidden = (feats.new_zeros((self.num_layers, batch_size, self.in_dim)),
                      feats.new_zeros((self.num_layers, batch_size, self.in_dim)))
            q_star = feats.new_zeros(batch_size, self.out_dim)
            for _ in range(self.num_iters):
                q, hidden = self.lstm(q_star.unsqueeze(1), hidden)
                q = q.view(batch_size, self.in_dim)
                e = (feats * dgl.broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata['e'] = e
                alpha = dgl.softmax_nodes(graph, 'e')
                graph.ndata['r'] = alpha * feats
                r = dgl.sum_nodes(graph, 'r')
                q_star = torch.cat([q, r], dim=-1)
                #a = nn.Softmax(dim=1)(e)
                #r = torch.sum(a * feats, dim=1, keepdim=True)
                #q_star = torch.cat((q, r), dim=2)
            #q_star = torch.squeeze(q_star, dim=1)
            #out = self.activation(self.predict(q_star))
        return q_star