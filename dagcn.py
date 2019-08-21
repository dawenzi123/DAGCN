from __future__ import print_function
import torch
import sys, os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib import GNNLIB
from pytorch_util import weights_init, gnn_spmm


class DAGCN(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats,
                 multi_h_emb_weight,
                 max_k=3, max_block=3, dropout=0.3, reg=1):
        print('Dual Attentional Graph Convolution')
        super(DAGCN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.dropout = dropout
        self.reg = reg
        self.multi_h_emb_weight = multi_h_emb_weight

        self.max_k = max_k
        self.max_block = max_block

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
            self.bne1 = nn.BatchNorm1d(latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)
            self.bne6 = nn.BatchNorm1d(output_dim)

        # Initial weights,
        self.conv_params = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for i in range(max_k))
        self.bn2 = nn.ModuleList(nn.BatchNorm1d(latent_dim) for i in range(max_k))

        # Node level attention
        self.k_weight = Parameter(torch.Tensor(self.max_k * latent_dim, latent_dim))
        self.bn3 = nn.BatchNorm1d(latent_dim)

        # Graph level attention
        self.att_params_w1 = nn.Linear(latent_dim, latent_dim)
        self.bn4 = nn.BatchNorm1d(latent_dim)

        self.att_params_w2 = nn.Linear(latent_dim, multi_h_emb_weight)
        self.bn5 = nn.BatchNorm1d(multi_h_emb_weight)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]

        dv = node_feat.device

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)
        n2n_sp, e2n_sp, subg_sp = n2n_sp.to(dv), e2n_sp.to(dv), subg_sp.to(dv)

        node_degs = 0

        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        h = self.attention_gcn(node_feat, edge_feat, n2n_sp, e2n_sp, node_degs)

        h = self.attention_pooling(h, subg_sp, graph_sizes)

        reg_term = self.get_reg(self.reg)

        return h, (reg_term/subg_sp.size()[0])

    def attention_gcn(self, node_feat, edge_feat, n2n_sp, e2n_sp, node_degs):
        input_node_linear = self.w_n2l(node_feat)
        input_node_linear = self.bn1(input_node_linear)

        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = self.bne1(input_edge_linear)

            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        input_potential = F.relu(input_message)

        block=0
        cur_message_layer = input_potential
        A = n2n_sp

        while block < self.max_block:
            if block == 0:
                block_input = cur_message_layer
            else:
                block_input = cur_message_layer + input_potential
            h = self.multi_hop_embedding(block_input, A, node_degs, input_message)
            h = F.relu(h)
            cur_message_layer = h
            block += 1

        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = cur_message_layer

        return reluact_fp

    def multi_hop_embedding(self, cur_message_layer, A, node_degs, input_message):
        step = 0
        input_x = cur_message_layer
        n, m = cur_message_layer.shape
        result = torch.zeros((n, m * self.max_k)).to(A.device)
        while step < self.max_k:
            n2npool = gnn_spmm(A, input_x) + cur_message_layer  # Y = (A + I) * X
            input_x = self.conv_params[step](n2npool)  # Y = Y * W
            input_x = self.bn2[step](input_x)
            result[:,(step * self.latent_dim):((step + 1) * self.latent_dim)] = input_x[:,:]
            step += 1

        return self.bn3(torch.matmul(result, self.k_weight).view(n, -1))

    def attention_pooling(self, node_emb, subg_sp, graph_sizes):

        atten_layer1 = self.bn4(torch.tanh(self.att_params_w1(node_emb)))
        atten_layer2 = self.bn5(self.att_params_w2(atten_layer1))

        graph_emb = torch.zeros(len(graph_sizes),self.multi_h_emb_weight, self.latent_dim)
        graph_emb = graph_emb.to(node_emb.device)
        graph_emb = Variable(graph_emb)

        accum_count = 0
        for i in range(subg_sp.size()[0]):
            alpha = atten_layer2[accum_count: accum_count + graph_sizes[i]]
            alpha = F.softmax(alpha, dim=-1)

            alpha = F.dropout(alpha, self.dropout)
            alpha = alpha.t()

            input_before = node_emb[accum_count: accum_count + graph_sizes[i]]
            emb_g = torch.matmul(alpha, input_before)

            graph_emb[i] = emb_g
            accum_count += graph_sizes[i]

        y_potential = graph_emb.view(len(graph_sizes), -1)

        return F.relu(y_potential)

    def get_reg(self, r=None):
        reg = 0

        for p in self.parameters():
            if p.dim() > 1:
                if r == 1:
                    reg += abs(p).sum()
                elif r == 2:
                    reg += p.pow(2).sum()

        return reg
