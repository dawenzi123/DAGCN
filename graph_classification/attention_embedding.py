from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from s2v_lib import S2VLIB
from pytorch_util import weights_init, gnn_spmm


class AttentionEmbedMeanField(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats,
                 multi_h_emb_weight,
                 max_k=3, max_block=3, dropout=0.3, reg=1):
        print('Attentional Embedding')
        super(AttentionEmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.dropout = dropout
        self.reg = reg
        self.multi_h_emb_weight = multi_h_emb_weight


        self.max_k = max_k
        self.max_block = max_block

        self.k_hop_embedding = True
        self.graph_level_attention = True

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)    # layer 1, linear for input X-->latent size
        self.bn1 = nn.BatchNorm1d(latent_dim)

        if num_edge_feats > 0:                                    # fixme: never been used
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)    # layer1.5, if edge_feats is not none, from X.edges --> latent
            self.bne1 = nn.BatchNorm1d(latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)   # layer 6 final layer of model
            self.bne6 = nn.BatchNorm1d(output_dim)

        # two ways to initial weights,
        # a) multiple convolution layer share the weights.
        # self.conv_params = nn.Linear(latent_dim, latent_dim)  # layer 2, graph convolution between k
        # self.bn2 = nn.BatchNorm1d(latent_dim)

        #b> every convolution layer have their own weights.
        self.conv_params = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for i in range(max_k))
        self.bn2 = nn.ModuleList(nn.BatchNorm1d(latent_dim) for i in range(max_k))

        # Node level attention
        self.k_weight = Parameter(torch.Tensor(self.max_k * latent_dim, latent_dim))    #layer 3, node level attention
        self.bn3 = nn.BatchNorm1d(latent_dim)

        # Graph level attention
        self.att_params_w1 = nn.Linear(latent_dim, latent_dim)      #layer 4
        self.bn4 = nn.BatchNorm1d(latent_dim)

        self.att_params_w2 = nn.Linear(latent_dim, multi_h_emb_weight)               #layer 5   both for self-attention.
        self.bn5 = nn.BatchNorm1d(multi_h_emb_weight)


        print('K hop convolutional:', self.k_hop_embedding)
        print('graph_level_attention:', self.graph_level_attention)
        print("Max_block", self.max_block)
        print("Dropout:", self.dropout)
        print("max_k:", self.max_k)
        print("multi_head:", self.multi_h_emb_weight)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        # node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        # node_degs = torch.cat(node_degs).unsqueeze(1)

        dv = node_feat.device

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        n2n_sp, e2n_sp, subg_sp = n2n_sp.to(dv), e2n_sp.to(dv), subg_sp.to(dv)
        # node_degs = node_degs.to(dv)

        node_degs = 0

        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        # node_degs = Variable(node_degs)

        h = self.node_level_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, node_degs)

        h = self.graph_level_embedding(h, subg_sp, graph_sizes)

        # return regular term
        reg_term = self.get_reg(self.reg)

        return h, (reg_term/subg_sp.size()[0])

    def node_level_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, node_degs):
        input_node_linear = self.w_n2l(node_feat)  # Question: could try to remove this layer. -->> this is for channels matching, hard to remove.
        input_node_linear = self.bn1(input_node_linear)

        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = self.bne1(input_edge_linear)

            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        input_potential = F.relu(input_message)

        if self.k_hop_embedding:
            block=0
            cur_message_layer = input_potential
            A = n2n_sp  # .to_dense() # 1 hop information

            while block < self.max_block:
                if block == 0:
                    block_input = cur_message_layer
                else:
                    block_input = cur_message_layer + input_potential
                h = self.multi_hop_embedding(block_input, A, node_degs, input_message)
                h = F.relu(h)       # fixme: do we need this relu after the block, may not.
                cur_message_layer = h
                block += 1

        else:  # simple aggregate the node features from neighbors, the same as structure2vec
            lv = 0
            cur_message_layer = input_potential
            while lv < self.max_lv:
                n2npool = gnn_spmm(n2n_sp, cur_message_layer)
                node_linear = self.conv_params(n2npool)   #layer3
                self.bn2 = nn.BatchNorm1d(latent_dim)
                merged_linear = node_linear + input_message

                cur_message_layer = F.relu(merged_linear)
                lv += 1

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
            result[:,(step * self.latent_dim):((step + 1) * self.latent_dim)] = input_x[:,:]  # fixme: 0 or 1 depend on element wise or layer wise.
            step += 1

        # return self.
        # h = h.div(node_degs)  # Y = D^-1 * Y

        return self.bn3(torch.matmul(result, self.k_weight).view(n, -1))

    def graph_level_embedding(self, node_emb, subg_sp, graph_sizes):

        if self.graph_level_attention:
            # Attention layer for nodes
            atten_layer1 = self.bn4(F.tanh(self.att_params_w1(node_emb)))
            atten_layer2 = self.bn5(self.att_params_w2(atten_layer1))

            graph_emb = torch.zeros(len(graph_sizes),self.multi_h_emb_weight, self.latent_dim)
            graph_emb = graph_emb.to(node_emb.device)
            graph_emb = Variable(graph_emb)

            accum_count = 0
            for i in range(subg_sp.size()[0]):
                alpha = atten_layer2[accum_count: accum_count + graph_sizes[i]]  # nodes in a single graphs
                # alpha = self.leakyrelu(alpha)
                alpha = F.softmax(alpha, dim=-1)  # softmax for normalization    bs[32, 8] --> [node_num, multi_head_channel]

                alpha = F.dropout(alpha, self.dropout)
                alpha = alpha.t()

                input_before = node_emb[accum_count: accum_count + graph_sizes[i]]  #vs[32, 64] --> [node_num, latent_dim]
                emb_g = torch.matmul(alpha, input_before)  # attention: alpha * h, a single row   bs[8,64] -->> [multi_head_channel, latent_dim]

                # emb_g = emb_g.view(1, -1)
                graph_emb[i] = emb_g      # bs[graph_num, multi_head_channel, latent_dim]
                accum_count += graph_sizes[i]

            y_potential = graph_emb.view(len(graph_sizes), -1)

        else:  # average aggregator
            y_potential = gnn_spmm(subg_sp, node_emb)

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
