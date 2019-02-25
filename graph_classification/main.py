import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
# from torch.autograd import Variable
# from torch.nn.parameter import Parameter
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

sys.path.append('%s/../s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from attention_embedding import AttentionEmbedMeanField
from embedding import EmbedMeanField, EmbedLoopyBP
from mlp import MLPClassifier
import pickle
from util import cmd_args, load_data
import datetime

def print2file(buf, outFile, p=False):
    if p:
        print(buf)
    outfd = open(outFile, 'a+')
    outfd.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
    outfd.close()


def environment_initial():
    # fixme: need a clean up
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if os.path.isfile(cmd_args.name):
        epoch_records = pickle.load(open(cmd_args.name, "rb"))
        epoch_records["fold"] += 1

        if epoch_records["fold"] == cmd_args.fold:
            pickle.dump(epoch_records, open(cmd_args.name, "wb"))
            return

    epoch_records = {"fold":1,
                         0:0,
                         100:0,
                         200:0,
                         300:0,
                         400:0,
                         500:0}

    pickle.dump(epoch_records, open(cmd_args.name, "wb"))


def k_fold_log(name, epoch, test_acc):
    epoch_records = pickle.load(open(cmd_args.name, "rb"))
    epoch_records[epoch] += test_acc
    pickle.dump(epoch_records, open(cmd_args.name, "wb"))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        model = AttentionEmbedMeanField

        self.s2v = model(latent_dim=cmd_args.latent_dim,
                         output_dim=cmd_args.out_dim,
                         num_node_feats=cmd_args.feat_dim,
                         num_edge_feats=0,
                         multi_h_emb_weight=cmd_args.multi_h_emb_weight,
                         max_k=cmd_args.max_k,
                         dropout=cmd_args.dropout,
                         max_block=cmd_args.max_block,
                         reg=cmd_args.reg)

        out_dim = cmd_args.out_dim

        if out_dim == 0:
            out_dim = cmd_args.latent_dim * cmd_args.multi_h_emb_weight

        if cmd_args.gm == 'attention':   #Note: outdim is 0 by most of case, may not need
            out_dim = cmd_args.latent_dim * cmd_args.multi_h_emb_weight  # multi-head=3
        #   self.mlp = MLPClassifier1layer(input_size=out_dim, num_class=cmd_args.num_class)
        # else:
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        concat_feat = []
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            concat_feat += batch_graph[i].node_tags
        
        concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
        node_feat = torch.zeros(n_nodes, cmd_args.feat_dim)
        node_feat.scatter_(1, concat_feat, 1)

        if cmd_args.mode >= 0:
            node_feat = node_feat.cuda() 
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph): 
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed, reg = self.s2v(batch_graph, node_feat, None)
        
        return self.mlp(embed, labels, reg)


def loop_dataset(g_list, classifier, sample_idxes, train=True, optimizer=None, bsize=cmd_args.batch_size):
    if train:
        classifier.train()
    else:
        classifier.eval()
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch', leave=False)

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc = classifier(batch_graph)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()

        # loss = loss.data[0]
        loss = loss.item()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss


if __name__ == '__main__':
    if cmd_args.mode >= 0:
        os.environ['CUDA_VISIBLE_DEVICES']=str(cmd_args.mode)

    environment_initial()

    # train_graphs, valid_graphs, test_graphs = load_data()
    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))
    print2file(str(cmd_args), cmd_args.logDir, p=True)
    print2file(str(cmd_args.logDes), cmd_args.logDir)

    classifier = Classifier()

    if cmd_args.mode >= 0:
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)   #Note: any better optimizer?

    train_idxes = list(range(len(train_graphs)))

    best_valid_acc = test_acc = 0
    old_epoch_records = []

    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)

        train_loss, train_acc = loop_dataset(train_graphs, classifier, train_idxes, train=True, optimizer=optimizer)
        # print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, train_loss, train_acc))

        # valid_loss, valid_acc = loop_dataset(valid_graphs, classifier, list(range(len(valid_graphs))), train=False, )
        # print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss, test_acc))

        # if valid_acc > best_valid_acc:
        test_loss, test_acc = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))),train=False,)
            # best_valid_acc = valid_acc
        # print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss, test_acc))

        # print2file('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, '
        #       'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss, train_acc,
        #                                                  valid_acc, test_acc), cmd_args.logDir, p=True)
        print2file('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, '
                   'Test Acc: {:.7f}'.format(epoch, train_loss, train_acc, test_acc), cmd_args.logDir, p=True)

        # log_info = str(epoch) + "," + str(test_acc)# + "," + str(test_loss)
        # print2file(str(log_info), cmd_args.logDir)

        # if best_acc is None or test_acc > best_acc:
        #     best_loss = test_loss
        #     best_acc = test_acc
        #     best_epoch = epoch
        # if (epoch+1) % 5 == 0:
        #     for name, param in classifier.named_parameters():
        #         if param.requires_grad:
        #             print (name, param.data)

        if (epoch+1) % 100 == 0:
            old_epoch_records.append(test_acc)
            k_fold_log(cmd_args.name, epoch+1, test_acc)

    # Print out summary
    # print2file("Done--- Best result is at epoch %s : loss %s, acc %s" %(best_epoch, best_loss, best_acc), cmd_args.logDir, p=True)

    print2file("old_epoch records:", cmd_args.logDir)
    print2file(str(old_epoch_records), cmd_args.logDir)

    epoch_records = pickle.load(open(cmd_args.name, "rb"))
    if epoch_records["fold"] == 10:
        for key, value in epoch_records.iteritems():
            statement = str(key) + "-->" + str(value/10)
            print2file(statement, cmd_args.logDir, p=True)

