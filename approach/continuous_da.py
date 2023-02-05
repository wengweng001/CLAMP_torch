import time
import torch
import numpy as np
from argparse import ArgumentParser

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset

from copy import deepcopy

class Continous_DA_Appr:
    """Basic class for continous domain adaptation approaches"""

    def __init__(self, model, device, nepochs=100, pre_epochs=1, meta_updates=1, base_updates=1, 
                 lr=0.05, lr_a=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset1: ExemplarsDataset = None,
                 exemplars_dataset2: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.pre_epochs = pre_epochs
        self.meta_updates = meta_updates
        self.base_updates = base_updates
        self.lr1 = lr
        self.lr2 = lr_a
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset1 = exemplars_dataset1
        self.exemplars_dataset2 = exemplars_dataset2
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None

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

    def train(self, t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader):
        """Main train structure"""
        # self.pre_train_process(t, trn_src_loader, trn_tgt_loader)
        self.train_loop(t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader)
        self.post_train_process(t, trn_src_loader)

    def pre_train_process(self, t, trn_src_loader, trn_tgt_loader):
        """Runs before training all epochs of the task (before the train session)"""
        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    # def train_loop(self, t, trn_src_loader, trn_tgt_loader, val_src_loader, val_tgt_loader):
    #     """Contains the epochs loop"""
    #     lr = self.lr1
    #     best_loss = np.inf
    #     patience = self.lr_patience
    #     best_model = self.model.get_copy()

    #     self.optimizer, self.optimizer1, self.optimizer2 = self._get_optimizer()

    #     # add exemplars to train_loader
    #     if len(self.exemplars_dataset1) > 0 and t > 0:
    #         trn_loader = torch.utils.data.DataLoader(trn_src_loader + self.exemplars_dataset1,
    #                                                  batch_size=trn_loader.batch_size,
    #                                                  shuffle=True,
    #                                                  num_workers=trn_loader.num_workers,
    #                                                  pin_memory=trn_loader.pin_memory)
    #     if len(self.exemplars_dataset2) > 0 and t > 0:
    #         trn_pseudo_loader = torch.utils.data.DataLoader(trn_tgt_loader + self.exemplars_dataset2,
    #                                                  batch_size=trn_loader.batch_size,
    #                                                  shuffle=True,
    #                                                  num_workers=trn_loader.num_workers,
    #                                                  pin_memory=trn_loader.pin_memory)

    #     # Loop epochs
    #     for e in range(self.nepochs):
    #         # Train
    #         clock0 = time.time()
    #         self.train_source(t, trn_src_loader)
    #         self.train_domains(t, trn_src_loader, trn_tgt_loader)
    #         if self.pseudo_label:
    #             self.self_labelling(trn_tgt_loader)
    #             self.train_target(t, trn_tgt_loader)
    #         clock1 = time.time()
    #         if self.eval_on_train:
    #             train_loss, train_acc, _ = self.eval(t, trn_tgt_loader)
    #             clock2 = time.time()
    #             print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
    #                 e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
    #             self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
    #             self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
    #         else:
    #             print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

    #         # Valid
    #         clock3 = time.time()
    #         valid_loss, valid_acc, _ = self.eval(t, val_tgt_loader)
    #         clock4 = time.time()
    #         print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
    #             clock4 - clock3, valid_loss, 100 * valid_acc), end='')
    #         self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
    #         self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

    #         # Adapt learning rate - patience scheme - early stopping regularization
    #         if valid_loss < best_loss:
    #             # if the loss goes down, keep it as the best model and end line with a star ( * )
    #             best_loss = valid_loss
    #             best_model = self.model.get_copy()
    #             patience = self.lr_patience
    #             print(' *', end='')
    #         else:
    #             # if the loss does not go down, decrease patience
    #             patience -= 1
    #             if patience <= 0:
    #                 # if it runs out of patience, reduce the learning rate
    #                 lr /= self.lr_factor
    #                 print(' lr={:.1e}'.format(lr), end='')
    #                 if lr < self.lr_min:
    #                     # if the lr decreases below minimum, stop the training session
    #                     print()
    #                     break
    #                 # reset patience and recover best model so far to continue training
    #                 patience = self.lr_patience
    #                 self.optimizer.param_groups[0]['lr'] = lr
    #                 self.model.set_state_dict(best_model)
    #         self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
    #         self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
    #         print()
    #     self.model.set_state_dict(best_model)

    def train_loop(self, t, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader):
        """Contains the epochs loop"""
        lr = self.lr1
        best_loss = np.inf
        best_loss_warmup = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer, self.optimizer1, self.optimizer2 = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, e, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, val_src_loader[t])
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_tgt_loader[t])
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="domain confusion")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="domain confusion")

            # Adapt learning rate - patience scheme - early stopping regularization
            if e<self.warmup_epochs:
                if valid_loss < best_loss_warmup:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss_warmup = valid_loss
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
                        for layer in range(len(self.optimizer.param_groups)):
                            self.optimizer.param_groups[layer]['lr'] = lr
                        self.model.set_state_dict(best_model)
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="domain train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="domain train")
                print()
            else:
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
                        for layer in range(len(self.optimizer.param_groups)):
                            self.optimizer.param_groups[layer]['lr'] = lr
                        self.model.set_state_dict(best_model)
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()
        self.model.set_state_dict(best_model)


    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass
    
    def self_labelling(self, trn_loader):
        """Runs pseudo labelling method for the target loader (before the target domain training session)"""
        pass
    
    def train_epoch(self, t, e, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for (x_s, labels_s), (x_t, _) in zip(trn_src_loader, trn_tgt_loader):
            # compute output
            x = torch.cat((x_s, x_t), dim=0)

            # Forward current model
            y, f = self.model.classifier(x)
            y_s, y_t = y.chunk(2, dim=0)

            f_s, f_t = f.chunk(2, dim=0)
            
            cls_loss = torch.nn.functional.cross_entropy(y_s, labels_s)

            transfer_loss = self.model.domain_adv(y_s, f_s, y_t, f_t)
            domain_acc = self.model.domain_adv.domain_discriminator_accuracy
            loss = cls_loss + transfer_loss * self.trade_off

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def train_source(self, t, trn_loader):
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

    def train_domains(self, t, trn_src_loader, trn_tgt_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for (x_s, _), (x_t, _) in zip(trn_src_loader, trn_tgt_loader):
            # compute output
            x = torch.cat((x_s, x_t), dim=0)

            # Forward current model
            y, f = self.model.classifier(x)
            y_s, y_t = y.chunk(2, dim=0)

            f_s, f_t = f.chunk(2, dim=0)
            
            transfer_loss = self.model.domain_adv(y_s, f_s, y_t, f_t)
            domain_acc = self.model.domain_adv.domain_discriminator_accuracy
            loss = transfer_loss * self.trade_off

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def train_target(self, t, trn_tgt_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_tgt_loader:
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
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
