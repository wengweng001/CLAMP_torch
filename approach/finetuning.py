import torch
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

import numpy as np
from copy import deepcopy

class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        # if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
        #     # if there are no exemplars, previous heads are not modified
        #     params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        # else:
        #     params = self.model.parameters()
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            # trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
            #                                          batch_size=trn_loader.batch_size,
            #                                          shuffle=True,
            #                                          num_workers=trn_loader.num_workers,
            #                                          pin_memory=trn_loader.pin_memory)
            print('add previous samples')
            print(len(trn_loader.dataset))
            if hasattr(self.exemplars_dataset.dataset,"sub_indeces"):
                trn_loader.dataset.cur_task_indeces = trn_loader.dataset.sub_indeces
                trn_loader.dataset.sub_indeces = trn_loader.dataset.sub_indeces + self.exemplars_dataset.dataset.sub_indeces
            elif hasattr(self.exemplars_dataset.dataset,"index"):
                trn_loader.dataset.cur_task_indeces = trn_loader.dataset.index
                trn_loader.dataset.index = trn_loader.dataset.index + self.exemplars_dataset.dataset.index
                trn_loader.dataset.samples = np.concatenate((trn_loader.dataset.samples,self.exemplars_dataset.dataset.samples),0)
                trn_loader.dataset.targets = np.concatenate((trn_loader.dataset.targets, self.exemplars_dataset.dataset.targets),0)
            print(len(trn_loader.dataset))

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            list(range(sum(self.model.task_cls[:t]),sum(self.model.task_cls[:t+1]))),
            deepcopy(trn_loader.dataset))

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        targets = targets.long()
        # if self.all_out or len(self.exemplars_dataset) > 0:
        #     return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        if self.all_out or len(self.exemplars_dataset) > 0:
            cur_cls = list(range(self.model.task_cls[:t+1].cumsum(0)[-1]))
            outputs = outputs[:, cur_cls]
            return torch.nn.functional.cross_entropy(outputs, targets)
        if (self.active_classes is not None):
            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
            outputs = outputs[:, class_entries]
        return torch.nn.functional.cross_entropy(outputs, targets - self.model.task_offset[t])