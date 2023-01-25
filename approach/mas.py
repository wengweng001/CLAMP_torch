import torch
import itertools
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import numpy as np
import copy
import utils

class Appr(Inc_Learning_Appr):
    """Class implementing the Memory Aware Synapses (MAS) approach (global version)
    described in https://arxiv.org/abs/1711.09601
    Original code available at https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=1, alp=0.5, fi_num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.lamb = lamb
        self.alp = alp
        self.num_samples = fi_num_samples

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                           if p.requires_grad}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 3: lambda is the regularizer trade-off -- In original code: MAS.ipynb block [4]: lambda set to 1
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off  (default=%(default)s)')
        # Define how old and new importance is fused, by default it is a 50-50 fusion
        parser.add_argument('--alp', default=0.5, type=float, required=False,
                            help='MAS alp (default=%(default)s)')
        # Number of samples from train for estimating importance
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        # if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
        #     # if there are no exemplars, previous heads are not modified
        #     params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        # else:
        #     params = self.model.parameters()
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    # Section 4.1: MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, t, trn_loader):
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                      if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        # n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
        #     else (len(trn_loader.dataset) // trn_loader.batch_size)
        n_samples_batches = (self.num_samples // self.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // self.batch_size)
        # Do forward and backward pass to accumulate L2-loss gradients
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
            outputs = self.model.forward(images.to(self.device))
            # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
            # loss = torch.norm(torch.cat(outputs, dim=1), p=2, dim=1).mean()
            cur_cls = list(range(self.model.task_cls[:t+1].cumsum(0)[-1]))
            outputs = outputs[:,cur_cls]
            loss = torch.norm(outputs, p=2, dim=1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)
        # Eq. 2: divide by N total number of samples
        n_samples = n_samples_batches * self.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            # trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
            #                                          batch_size=trn_loader.batch_size,
            #                                          shuffle=True,
            #                                          num_workers=trn_loader.num_workers,
            #                                          pin_memory=trn_loader.pin_memory)
            if hasattr(self.exemplars_dataset.dataset,"sub_indeces"):
                trn_loader.dataset.cur_task_indeces = trn_loader.dataset.sub_indeces
                trn_loader.dataset.sub_indeces = trn_loader.dataset.sub_indeces + self.exemplars_dataset.dataset.sub_indeces
            elif hasattr(self.exemplars_dataset.dataset,"index"):
                trn_loader.dataset.cur_task_indeces = trn_loader.dataset.index
                trn_loader.dataset.index = trn_loader.dataset.index + self.exemplars_dataset.dataset.index
                trn_loader.dataset.samples = np.concatenate((trn_loader.dataset.samples,self.exemplars_dataset.dataset.samples),0)
                trn_loader.dataset.targets = np.concatenate((trn_loader.dataset.targets, self.exemplars_dataset.dataset.targets),0)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            list(range(sum(self.model.task_cls[:t]),sum(self.model.task_cls[:t+1]))),
            copy.deepcopy(trn_loader.dataset))

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        trn_loader = utils.get_data_loader(trn_loader, batch_size=self.batch_size, cuda=self.device,drop_last=True)

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_importance = self.estimate_parameter_importance(t, trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.importance.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alp
            if self.alp == -1:
                # alp = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                if t==0:
                    alp = (0.0/(self.model.task_cls[-1])).to(self.device)
                else:
                    alp = (self.model.task_cls[-2] / self.model.task_cls[-1] ).to(self.device)
                self.importance[n] = alp * self.importance[n] + (1 - alp) * curr_importance[n]
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance[n] = self.alp * self.importance[n] + (1 - self.alp) * curr_importance[n]

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            # Eq. 3: memory aware synapses regularizer penalty
            for n, p in self.model.model.named_parameters():
                if n in self.importance.keys():
                    loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
            loss += self.lamb * loss_reg
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            cur_cls = list(range(self.model.task_cls[:t+1].cumsum(0)[-1]))
            outputs = outputs[:,cur_cls]
        #     return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
            return loss + torch.nn.functional.cross_entropy(outputs, targets)
        # return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        if (self.active_classes is not None):
            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
            outputs = outputs[:, class_entries]
        return loss + torch.nn.functional.cross_entropy(outputs, targets - self.model.task_offset[t])
