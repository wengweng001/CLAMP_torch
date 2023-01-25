import torch
from argparse import ArgumentParser

from datasets.exemplars_dataset import ExemplarsDataset
from .domain_adaptation import dom_adapt_Appr

from utils import ForeverDataIterator, get_data_loader

import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torchvision.utils import save_image
from utils import ForeverDataIterator
from datasets.exemplars_dataset import ExemplarsDataset
import time
import utils
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import itertools
import os

class Appr(dom_adapt_Appr):
    """Implementing the Unsupervised Domain Adaptation by Backpropagation (DANN) approach
    described in http://sites.skoltech.ru/compvision/projects/grl/
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, logger,
                                   exemplars_dataset)


    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(),
                lr=self.lr, weight_decay=self.wd, 
                momentum=self.momentum
                )

    def train_loop(self, t, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader):
        """Contains the epochs loop"""
        
        self.total_steps = self.nepochs * len(trn_tgt_loader)

        super().train_loop(t, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader)

        # self.visualize(val_src_loader, val_tgt_loader, 'source', self.save_name)

    def pre_train_process(self, t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader):
        if self.warmup_epochs \
            and t > 0 \
            :
            classifier_criterion = nn.CrossEntropyLoss().to(self.device)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.warmup_lr, momentum=self.momentum)
            
            for epoch in range(self.warmup_epochs):
                warmupclock0 = time.time()
                print('Epoch : {}'.format(epoch))
                self.model.train()

                start_steps = epoch * len(trn_src_loader)
                total_steps = self.warmup_epochs * len(trn_tgt_loader)
                
                for batch_idx, (source_data, _) in enumerate(zip(trn_src_loader, trn_tgt_loader)):
                    source_image, source_label = source_data
                    p = float(batch_idx + start_steps) / total_steps

                    source_image, source_label = source_image.to(self.device), source_label.to(self.device)

                    # optimizer = self.optimizer_scheduler(optimizer=optimizer, p=p)
                    optimizer.zero_grad()

                    source_feature, _ = self.model(source_image)

                    # Classification loss
                    class_pred = self.model.classifier(source_feature)
                    if self.active_classes is not None:
                        class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                        class_pred = class_pred[:, class_entries]
                        source_label = source_label - class_entries[0]
                    class_loss = classifier_criterion(class_pred, source_label)

                    class_loss.backward()
                    optimizer.step()
                    if (batch_idx + 1) % 50 == 0:
                        print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image), len(trn_src_loader.dataset), 100. * batch_idx / len(trn_src_loader), class_loss.item()))

                warmupclock1 = time.time()
                
            trn_loss, trn_acc, _ = self.eval_(t, val_src_loader, val_tgt_loader, training_mode='source_only')
            warmupclock2 = time.time()
                
            print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                epoch + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
            self.logger.log_scalar(task=t, iter=epoch + 1, name="loss", value=trn_loss, group="warmup")
            self.logger.log_scalar(task=t, iter=epoch + 1, name="acc", value=100 * trn_acc, group="warmup")
            # self.visualize(val_src_loader, val_tgt_loader, 'source', self.save_name)

    def train_loop(self, t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader):
        
        self.optimizer = self._get_optimizer()
        classifier_criterion = nn.CrossEntropyLoss()
        discriminator_criterion = nn.CrossEntropyLoss()

        for epoch in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.model.train()

            len_dataloader = min(len(trn_src_loader), len(trn_tgt_loader))

            for batch_idx, (source_data, target_data) in enumerate(zip(trn_src_loader, trn_tgt_loader)):

                source_image, source_label = source_data
                target_image, target_label = target_data

                p = float(batch_idx + epoch * len_dataloader) / self.nepochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                source_image, source_label = source_image.to(self.device), source_label.to(self.device)
                target_image, target_label = target_image.to(self.device), target_label.to(self.device)
                # combined_image = torch.cat((source_image, target_image), 0)

                # self.optimizer = self.optimizer_scheduler(optimizer=self.optimizer, p=p)
                self.optimizer.zero_grad()


                domain_label = torch.zeros(len(source_image))
                domain_label = domain_label.long()
                a,b = self.model(source_image,alpha)
                err_s_domain = discriminator_criterion(b, domain_label.to(self.device))

                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    a = a[:, class_entries]
                    source_label = source_label - class_entries[0]

                # training model using target data
                domain_label = torch.ones(len(target_image))
                domain_label = domain_label.long()
                _, b = self.model(target_image,alpha)
                err_t_domain = discriminator_criterion(b, domain_label.to(self.device))

                err_s_label = classifier_criterion(a, source_label)
                total_loss = err_s_label + err_s_domain + err_t_domain
                total_loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 50 == 0:
                    print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                        batch_idx * len(target_image), len(trn_tgt_loader.dataset), 100. * batch_idx / len(trn_tgt_loader), total_loss.item(), err_s_label.item(), (err_s_domain + err_t_domain).item()))

            clock1 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(epoch + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval_(t, val_src_loader, val_tgt_loader, 'dann')
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=epoch + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=epoch + 1, name="acc", value=100 * valid_acc, group="valid")
            self.logger.log_scalar(task=t, iter=epoch + 1, name="patience", value=self.lr_patience, group="train")
            self.logger.log_scalar(task=t, iter=epoch + 1, name="lr", value=self.lr, group="train")

    def eval_(self, t, source_test_loader, target_test_loader, training_mode):
        # print("Model test ...")

        warmup_loss = torch.nn.CrossEntropyLoss()

        self.model.eval()

        test_loss = 0
        correct = 0
        total_num = 0
        with torch.no_grad():
            for data, target in source_test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data,0.5)
                test_loss += float(warmup_loss(output, target))  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += float(pred.eq(target.view_as(pred)).sum())
                total_num += len(target)
                
        test_loss /= len(source_test_loader.dataset)
        print('Test set (source): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, total_num,
            100. * correct / total_num))

        self.model.eval()
        test_loss = 0
        total_num = 0
        total_acc_taw = 0
        total_acc_tag = 0
        with torch.no_grad():
            for data, target in target_test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data,0.5)
                output_ = output
                target_  = target
                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    output_ = output[:, class_entries]
                    target_  = target-class_entries[0]
                test_loss += float(warmup_loss(output_, target_))  # sum up batch loss
                pred = output_.max(1, keepdim=True)[1]  # get the index of the max log-probability
                total_acc_taw += float(pred.eq(target_.view_as(pred)).sum())
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                total_acc_tag += float(pred.eq(target.view_as(pred)).sum())
                total_num += len(target)

        print('Test set (target): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, total_acc_taw, total_num,
            100. * total_acc_taw / total_num))
        
        return test_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
    
    def eval(self, t, dataset):
        warmup_loss = torch.nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            for data, target in dataset:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data,0.5)
                output_ = output
                target_  = target
                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    output_ = output[:, class_entries]
                    target_  = target-class_entries[0]
                total_loss += float(warmup_loss(output_, target_))  # sum up batch loss
                pred = output_.max(1, keepdim=True)[1]  # get the index of the max log-probability
                total_acc_taw += float(pred.eq(target_.view_as(pred)).sum())
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                total_acc_tag += float(pred.eq(target.view_as(pred)).sum())
                total_num += len(target)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
        
    def optimizer_scheduler(self, optimizer, p):
        """
        Adjust the learning rate of optimizer
        :param optimizer: optimizer for updating parameters
        :param p: a variable for adjusting learning rate
        :return: optimizer
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr / (1. + 10 * p) ** 0.75

        return optimizer

# def plot_embedding(X, y, d, training_mode, save_name):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#     y = np.array(y)

#     plt.figure(figsize=(10, 10))
#     for i in range(len(d)):  # X.shape[0] : 1024
#         # plot colored number
#         if d[i] == 0:
#             colors = (0.0, 0.0, 1.0, 1.0)
#         else:
#             colors = (1.0, 0.0, 0.0, 1.0)
#         plt.text(X[i, 0], X[i, 1], str(y[i]),
#                  color=colors,
#                  fontdict={'weight': 'bold', 'size': 9})

#     plt.xticks([]), plt.yticks([])
#     if save_name is not None:
#         plt.title(save_name)

#     save_folder = 'images/dann/'
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     fig_name = 'images/dann/' + str(training_mode) + '_' + str(save_name) + '.png'
#     plt.savefig(fig_name)
#     print('{} is saved'.format(fig_name))
