from __future__ import print_function
import os, sys
sys.path.append('./gcn')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np

from model.build_gen import *
from datasets.dataset_read import dataset_read_officehome, dataset_read_office31
from gcn.models import GCN

class Graph(nn.Module):
    def __init__(self, args):
        super(Graph, self).__init__()
        self.args = args
        # self.linear = torch.nn.Linear(args.embed, args.nclasses)
        self.mean = torch.nn.Parameter(torch.rand(args.nclasses + self.args.ndomain, args.embed).cuda()) # 
        self.adj = torch.exp(-self.euclid_dist(self.mean, self.mean) / (2 * self.args.sigma ** 2)) 
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist
    def forward(self):
        self.adj = torch.exp(-self.euclid_dist(self.mean, self.mean) / (2 * self.args.sigma ** 2)) 

# The solver for training and testing SSG
class Solver(object):
    def __init__(self, args, batch_size=16,
                 target='clipart', learning_rate=0.0002, interval=10, optimizer='adam',
                 checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.interval = interval
        self.lr = learning_rate
        self.best_correct = 0
        self.args = args
        self.beta = torch.distributions.Beta(0.1, 0.1)
        if self.args.use_target:
            self.ndomain = self.args.ndomain
        else:
            self.ndomain = self.args.ndomain - 1

        # load source and target domains
        self.datasets, self.dataset_test, self.dataset_size = dataset_read_office31(target, self.batch_size)
        self.niter = self.dataset_size / self.batch_size
        print('Dataset loaded!')

        # define the feature extractor and GCN-based classifier
        self.G = Generator(self.args.net)
        self.GCN = GCN(nfeat=args.embed, nclasses=args.nfeat)
        self.G.cuda()
        self.GCN.cuda()
        print('Model initialized!')

        if self.args.load_checkpoint is not None:
            self.state = torch.load(self.args.load_checkpoint)
            self.G.load_state_dict(self.state['G'])
            self.GCN.load_state_dict(self.state['GCN'])
            print('Model load from: ', self.args.load_checkpoint)

        # initialize statistics (prototypes and adjacency matrix)
        if self.args.load_checkpoint is None:
            self.graphs = Graph(args = args)
            print('Statistics initialized!')
        else:
            self.graphs.mean = self.state['mean'].cuda()
            self.graphs.adj = self.state['adj'].cuda()
            print('Statistics loaded!')

        # define the optimizer
        self.set_optimizer(which_opt=optimizer, lr=self.lr)
        print('Optimizer defined!')

    # optimizer definition
    def set_optimizer(self, which_opt='sgd', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_gcn = optim.SGD(self.GCN.parameters(),
                                     lr=lr, weight_decay=0.0005,
                                     momentum=momentum)
            self.opt_graphs = optim.SGD(self.graphs.parameters(),
                                     lr=lr, weight_decay=0.005,
                                     momentum=momentum)
        elif which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_gcn = optim.Adam(self.GCN.parameters(),
                                      lr=lr, weight_decay=0.0005)
            self.opt_graphs = optim.Adam(self.graphs.parameters(),
                                     lr=lr, weight_decay=0.005)

    # empty gradients
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_gcn.zero_grad()
        self.opt_graphs.zero_grad()

    # compute the discrepancy between two probabilities
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist

    # construct the extended adjacency matrix
    def construct_adj(self, feats):
        dist = self.euclid_dist(self.graphs.mean, feats)
        sim = torch.exp(-dist / (2 * self.args.sigma ** 2))
        E = torch.eye(feats.shape[0]).float().cuda()

        A = torch.cat([self.graphs.adj, sim], dim = 1)
        B = torch.cat([sim.t(), E], dim = 1)
        gcn_adj = torch.cat([A, B], dim = 0)

        return gcn_adj

    # assign pseudo labels to target samples
    def pseudo_label(self, logit, feat):
        pred = F.softmax(logit, dim=1)
        entropy = (-pred * torch.log(pred)).sum(-1)
        label = torch.argmax(logit, dim=-1).long()

        mask = (entropy < self.args.entropy_thr).float() 
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat, 0, index)
        label_ = torch.index_select(label, 0, index)

        return feat_, label_



    def get_label_condition(self, domain_label, category_label, zero_prob = 0.5, zero_prob_class = 0.5):
        
        label_condition = []
        domain_label = list(domain_label.cpu().numpy())
        category_label = list(category_label.cpu().numpy())
        for i in range(len(domain_label)):
            if len(category_label) < len(domain_label):
                category_label.append(self.args.nclasses + self.ndomain + 10)
            label_condition_batch = []
            mask_classes = np.random.choice([0,1], (self.args.nclasses), p = [zero_prob_class, 1.0 - zero_prob_class])
            mask_domains = np.random.choice([0,1], (self.ndomain), p = [zero_prob, 1.0 - zero_prob])
            mask = np.append(mask_classes, mask_domains) # np.random.choice([0,1], (self.args.nclasses + self.ndomain), p = [zero_prob, 1.0 - zero_prob])
            for j in range(len(mask)):
                tem = np.ones(self.args.nfeat) * mask[j]
                if j < self.args.nclasses:
                    if category_label[i] == j:
                        tem *= 0.0
                    else:
                        tem *= -0.0
                else:
                    if domain_label[i] == j - self.args.nclasses:
                        tem *= 0.1
                    else:
                        tem *= -0.1
                label_condition_batch.append(tem)
            label_condition.append(label_condition_batch)
        label_condition = np.array(label_condition)

        return label_condition


    # per epoch training in a Multi-Source Domain Adaptation setting
    def train_gcn_adapt(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.GCN.train()

        for batch_idx, data in enumerate(self.datasets):
            img_s = list()
            label_s = list()
            self.graphs.adj = torch.exp(-self.euclid_dist(self.graphs.mean, self.graphs.mean) / (2 * self.args.sigma ** 2)) 
            stop_iter = False
            for domain_idx in range(self.ndomain - 1): 
                tmp_img = data['S' + str(domain_idx + 1)].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

                if tmp_img.size()[0] < self.batch_size:
                    stop_iter = True

            if stop_iter:
                break

            # get the target batch
            img_t = data['T'].cuda()
            if img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()

            labels = torch.cat(label_s, dim=0)
            domain_label = []
            for domain_idx in range(self.ndomain):
                domain_label.append(torch.tensor([domain_idx] * self.args.batch_size))
            domain_label = torch.cat(domain_label, dim = 0)

            # get feature embeddings
            feat_list = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_img = img_s[domain_idx] 
                tmp_feat = self.G(tmp_img) 
                feat_list.append(tmp_feat)

            feat_t = self.G(img_t)
            feat_list.append(feat_t)
            feats = torch.cat(feat_list, dim=0)

            label_condition = self.get_label_condition(domain_label, labels, 0.5) 
            # add query samples to the domain graph
            gcn_feats = self.graphs.mean
            

            gcn_adj = self.graphs.adj.detach() 
            # output classification logit with GCN

            gcn_logit = self.GCN(gcn_feats, gcn_adj)

            gcn_logit = gcn_logit.repeat(len(domain_label), 1, 1) 
            label_condition = torch.from_numpy(label_condition).to(torch.float32).cuda()
            gcn_logit = gcn_logit + label_condition
            feats = torch.unsqueeze(feats, 1)


            gcn_logits = torch.matmul(feats, gcn_logit.transpose(1, 2)) # 640 * 50 checked
            gcn_logits = torch.squeeze(gcn_logits, dim = 1)

            feat_t_, label_t_ = self.pseudo_label(gcn_logits[-feat_t.shape[0]:, :self.args.nclasses], feat_t)
            feat_list.pop()
            feat_list.append(feat_t_)
            label_s.append(label_t_)

            # define GCN classification losses

            domain_label = domain_label.long().cuda()

            loss_cls_src = criterion(gcn_logits[:-feat_t.shape[0], :self.args.nclasses], labels) ## CE loss class

            target_logit = gcn_logits[-feat_t.shape[0]:, :self.args.nclasses] ####
            target_prob = F.softmax(target_logit, dim=1) ##### sigmoid? 
            loss_cls_tgt = (-target_prob * torch.log(target_prob + 1e-8)).mean() ## loss target

            loss_dom_tgt = criterion(gcn_logits[:, self.args.nclasses:], domain_label) 
            loss_cls = loss_cls_src + 5 * loss_cls_tgt + 0.1 * loss_dom_tgt # + loss_cls_dom ## loss class 

            # define relation alignment losse

            loss = loss_cls 

            # back-propagation
            loss.backward(retain_graph = True)
            self.opt_gcn.step()
            self.opt_g.step()
            self.opt_graphs.step()

            # record training information
            if epoch ==0 and batch_idx==0:
                record = open(record_file, 'a')
                record.write(str(self.args)+'\n')
                record.close()

            if batch_idx % self.interval == 0:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_source: {:.5f}'
                    '\tLoss_cls_target: {:.5f}\t loss_dom_tgt: {:.5f}\t'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                        loss_cls_src.item(), loss_cls_tgt.item(), loss_dom_tgt.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_source: {:.5f}'
                        '\tLoss_cls_target: {:.5f}\t loss_dom_tgt: {:.5f}\t'.format(
                            epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                            loss_cls_src.item(), loss_cls_tgt.item(), loss_dom_tgt.item()))
                    record.close()

        return batch_idx

    # per epoch test on target domain
    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.GCN.eval()
        test_loss = 0
        correct = 0
        size = 0

        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()

            feat = self.G(img)

            labels = torch.ones(len(feat)) * (self.args.nclasses + 1) 
            domain_label = torch.ones(len(feat)) * self.args.ndomain # default is the target domain
            label_condition = self.get_label_condition(domain_label, labels, 0.0)

            feat = torch.unsqueeze(feat, 1)
            gcn_feats = self.graphs.mean 
            gcn_adj = self.graphs.adj.detach()

            gcn_logit = self.GCN(gcn_feats, gcn_adj) 
            gcn_logit = gcn_logit.repeat(len(domain_label),1,1)
            label_condition = torch.from_numpy(label_condition).to(torch.float32).cuda()
            gcn_logit = gcn_logit + label_condition

            gcn_logits = torch.matmul(feat, gcn_logit.transpose(1, 2))
            gcn_logits = torch.squeeze(gcn_logits, dim = 1)
            gcn_logits_class = gcn_logits[:, :self.args.nclasses] 
            output = gcn_logits_class

            test_loss += -F.nll_loss(output, label).item()
            pred = output.max(1)[1]
            k = label.size()[0]
            correct += pred.eq(label).cpu().sum()
            size += k

        test_loss = test_loss / size

        if correct > self.best_correct:
            self.best_correct = correct
            if save_model:
                best_state = {'G': self.G.state_dict(), 'GCN': self.GCN.state_dict(), 'mean': self.graphs.mean.cpu(),
                              'adj': self.graphs.adj.cpu(), 'epoch': epoch}
                torch.save(best_state, os.path.join(self.checkpoint_dir, 'best_model.pth'))

        # save checkpoint
        if save_model and epoch % self.save_epoch == 0:
            state = {'G': self.G.state_dict(), 'GCN': self.GCN.state_dict(), 'mean': self.graphs.mean.cpu(),
                     'adj': self.graphs.adj.cpu()}
            torch.save(state, os.path.join(self.checkpoint_dir, 'epoch_' + str(epoch) + '.pth'))

        # record test information
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Best Accuracy: {}/{} ({:.4f}%)  \n'.format(
                test_loss, correct, size, 100. * float(correct) / size, self.best_correct, size,
                                          100. * float(self.best_correct) / size))

        if record_file:
            if epoch == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()

            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write(
                '\nEpoch {:>3} Average loss: {:.5f}, Accuracy: {:.5f}, Best Accuracy: {:.5f}'.format(
                    epoch, test_loss, 100. * float(correct) / size, 100. * float(self.best_correct) / size))
            record.close()