import time
import torch
import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset

from utils import get_data_loader
from copy import deepcopy

class dom_adapt_Appr:
    """Basic class for continous domain adaptation approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.optimizer = None
        self.active_classes = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_src, val_src, trn_tgt, val_tgt):
        """Main train structure"""
        trn_src_loader = get_data_loader(trn_src, batch_size=self.batch_size, cuda=self.device)
        val_src_loader = get_data_loader(val_src, batch_size=self.batch_size, cuda=self.device)
        trn_tgt_loader = get_data_loader(trn_tgt, batch_size=self.batch_size, cuda=self.device)
        val_tgt_loader = get_data_loader(val_tgt, batch_size=self.batch_size, cuda=self.device)
        self.pre_train_process(t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader)
        self.train_loop(t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader)
        self.post_train_process(t, trn_tgt_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        # Warm-up phase
        if self.warmup_epochs and t > 0:
            # self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            self.optimizer = torch.optim.SGD(self.model.heads.parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                # self.model.heads[-1].train()
                self.model.heads.train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    # loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    if self.active_classes is not None:
                        class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                        outputs = outputs[:, class_entries]
                    loss = self.warmup_loss(outputs, targets.to(self.device))
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    torch.nn.utils.clip_grad_norm_(self.model.heads.parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        # loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        if self.active_classes is not None:
                            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                            outputs = outputs[:, class_entries]
                        loss = self.warmup_loss(outputs, targets.to(self.device))
                        pred = torch.zeros_like(targets.to(self.device))
                        # for m in range(len(pred)):
                        #     this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                        #     pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        for m in range(len(pred)):
                            pred[m] = outputs[m].argmax()
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                # total_num = len(trn_loader.dataset.labels)
                total_num = len(trn_loader.dataset)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_src_loader, trn_tgt_loader)
            clock1 = time.time()
            print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_src_loader, val_tgt_loader, 'dann')
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Single-Head
        if (self.active_classes is not None) and len(self.exemplars_dataset) == 0:
            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
            outputs = outputs[:, class_entries]
        for m in range(len(pred)):
            pred[m] = outputs[m].argmax()
        # # Task-Aware Multi-Head
        # for m in range(len(pred)):
        #     this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
        #     pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # # Task-Agnostic Multi-Head
        # if self.multi_softmax:
        #     outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
        #     pred = torch.cat(outputs, dim=1).argmax(1)
        # else:
        #     pred = torch.cat(outputs, dim=1).argmax(1)
        pred = outputs.argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        # return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        if (self.active_classes is not None):
            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
            outputs = outputs[:, class_entries]
            targets = targets - class_entries[0]
        return torch.nn.functional.cross_entropy(outputs[t], targets)
