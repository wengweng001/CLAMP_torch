from audioop import mul
from logging import raiseExceptions
from pickle import FALSE, TRUE
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from re import T
from tabnanny import verbose
import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torchvision.utils import save_image
from utils import ForeverDataIterator
from .continuous_da import Continous_DA_Appr
from datasets.exemplars_dataset import ExemplarsDataset
import copy
import time
import utils
import os
from imblearn.under_sampling import RandomUnderSampler
from torchvision import transforms
import torch.nn.functional as F
import evaluate

from datasets.memory_dataset import MemoryDataset
from torch.utils.data import TensorDataset

class Appr(Continous_DA_Appr):
    """Class implementing the CLAMP approach"""

    def __init__(self, model, device, nepochs=100, pre_epochs=1, meta_updates=1, base_updates=1, 
                 lr=0.05, lr_a=0.01, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset1=None, exemplars_dataset2=None,
                 pseudo=False, meta=False, domain_inv=False, alpha=1.0):
        super(Appr, self).__init__(model, device, nepochs, pre_epochs, meta_updates, base_updates, 
                                   lr, lr_a, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset1, exemplars_dataset2)
        self.model_old          = None
        # module switch
        self.meta               = meta
        self.domain_invariant   = domain_inv
        self.pseudo             = pseudo
        # hyper param
        self.trade_off          = 1.0
        self.alpha              = alpha
        self.pseudo_threshold   = 0.85
        self.thresholds          = []
        # function switch
        self.der                = True
        self.distill            = True
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    # @staticmethod
    # def extra_parser(args):
    #     """Returns a parser containing the approach specific parameters"""
    #     parser = ArgumentParser()
    
    #     return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        finetune = False
        params = [
            {"params": self.model.classifier.backbone.parameters(), "lr": 0.1 * self.lr1 if finetune else 1.0 * self.lr1},
            {"params": self.model.classifier.pool_layer.parameters(), "lr": 1.0 * self.lr1},
            {"params": self.model.classifier.bottleneck.parameters(), "lr": 1.0 * self.lr1},
            {"params": self.model.classifier.head.parameters(), "lr": 1.0 * self.lr1},
            {"params": self.model.domain_adv.parameters(), "lr":  1.0 * self.lr1}]

        params_s = list(self.model.assessorS.parameters())
        params_t = list(self.model.assessorT.parameters())
        
        return torch.optim.SGD(params, lr=self.lr1, weight_decay=self.wd, momentum=self.momentum),\
            torch.optim.SGD(params_s, lr=self.lr2, weight_decay=self.wd, momentum=self.momentum),\
                torch.optim.SGD(params_t, lr=self.lr2, weight_decay=self.wd, momentum=self.momentum),

    def generate_transformed_dataset(self, t, dataset, balance_data=True):
        
        notSaved=True
        isEmpty=True
        for x, y in dataset:
            x0 = x
            #Gaussian noise
            x = x + (0.1**0.5)*torch.randn(x.shape)
            #Random RGB
            if x.shape[0]==28:
                x = x.unsqueeze(0).unsqueeze(0)
            x = transforms.functional.rotate(x,5)
            #Invery
            x = transforms.functional.invert(x)
            
            if(notSaved):
                p_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
                p_dir = os.path.join(p_dir, 'images')
                p_dir = os.path.join(p_dir, '3xtrf_val_src_task{}.jpg'.format(t+1))
                save_image(torch.cat((x0.unsqueeze(0),x.unsqueeze(0)),0),p_dir)
                notSaved=False

            if(isEmpty):
                if len(x.shape) == 3:
                    data = x.unsqueeze(0)
                if len(y.shape) == 0:
                    y = y.unsqueeze(0)
                targets = y.cpu()
                isEmpty=False
            else:
                if len(x.shape) == 3:
                    x = x.unsqueeze(0)
                data = torch.cat((data, x),dim=0)
                if len(y.shape) == 0:
                    y = y.unsqueeze(0)
                targets = torch.cat((targets, y.cpu()),dim=0)
        x_rus = data
        y_rus = targets

        if balance_data:
            rus = RandomUnderSampler(random_state=0)
            x_res, y_res = rus.fit_resample(data.view(data.shape[0],-1).numpy(), targets.numpy())
            if x_res.shape[1]==3*224*224:
                x_res = x_res.reshape(x_res.shape[0],3,224,224)
            if x_res.shape[1]==784:
                x_res = x_res.reshape(x_res.shape[0],1,28,28)
            x_rus = torch.from_numpy(x_res)
            y_rus = torch.from_numpy(y_res).long()

        from torch.utils.data import TensorDataset
        trans_dataset = TensorDataset(x_rus,y_rus)

        return trans_dataset

    def generate_transformed_samples(self, x, y):
        torch.cuda.empty_cache()
        notSaved=True
        isEmpty=True
        
        for (images, targets) in zip(x, y):
            x1 = images.detach().cpu()
            #Gaussian noise
            x1 = x1 + (0.1**0.5)*torch.randn(x1.shape)
            #Random RGB
            if x1.shape[0]==28:
                x1 = x1.unsqueeze(0).unsqueeze(0)
            x1 = transforms.functional.rotate(x1,5)
            #Invery
            x1 = transforms.functional.invert(x1)

            if(isEmpty):
                trf_imgs_x = x1.unsqueeze(0)
                trf_imgs_y = targets.unsqueeze(0).cpu()
                isEmpty=False
            else:
                trf_imgs_x = torch.cat((trf_imgs_x, x1.unsqueeze(0)),dim=0)
                trf_imgs_y = torch.cat((trf_imgs_y, targets.unsqueeze(0).cpu()),dim=0)

        # print("Random Transformation Finish!!")
        # print(trf_imgs_x.shape)
        # print(trf_imgs_y.shape)

        return trf_imgs_x, trf_imgs_y

    def generate_val_tgt(self, t, dataset, multiple_transform=False):
        '''Generate target validation samples -- domain confusion samples

        [src_dataloader]               source dataset'''
        torch.cuda.empty_cache()
        
        top_n = 2
        dataloader = utils.get_data_loader(dataset, self.batch_size, cuda=self.device, drop_last=True)
        diff_buffer = []
        sample_buffer = []
        target_buffer = []
        val_T_buffer_x = []
        val_T_buffer_y = []

        with torch.no_grad():
            self.model.domain_adv.eval()
            for image, target in dataloader:

                _, features = self.model.classifier(image.to(self.device))
                # domain_output = self.model.domain_adv.domain_discriminator(features)
                # diff = torch.abs(domain_output-0.5)
                domain_output = self.model.domain_adv(features, alpha=0.5)
                diff = torch.abs(F.softmax(domain_output)[:,0]-F.softmax(domain_output)[:,1])
                
                [diff_buffer.append(i.item()) for i in diff]
                sample_buffer.append(image)
                target_buffer.append(target)
        self.model.domain_adv.train()

        sample_buffer = torch.cat(sample_buffer)
        target_buffer = torch.cat(target_buffer)
        i_cls = torch.unique(target_buffer)
        for c in i_cls:
            ind = (target_buffer == c).nonzero(as_tuple=True)[0]
            ind = ind[np.argsort(np.array(diff_buffer)[ind])[:top_n]]
            val_T_buffer_x.append(sample_buffer[ind])
            val_T_buffer_y.append(target_buffer[ind])
        val_T_buffer_x = torch.cat(val_T_buffer_x,dim=0)
        val_T_buffer_y = torch.cat(val_T_buffer_y,dim=0)
        
        if multiple_transform:
            val_T_buffer_x,val_T_buffer_y = self.generate_transformed_samples(val_T_buffer_x,val_T_buffer_y)
            
        val_set = TensorDataset(val_T_buffer_x,val_T_buffer_y)
        p_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
        p_dir = os.path.join(p_dir, 'images')
        p_dir = os.path.join(p_dir, '3xtrf_val_tgt_task{}.jpg'.format(t+1))
        save_image(val_T_buffer_x, p_dir)
        torch.cuda.empty_cache()
        # return val_set
        return val_T_buffer_x,val_T_buffer_y
    
    def generate_pseudo_labels_by_threshold(self, dataset, active_classes=None):
        '''Generate target pseudo samples

        [tgt_dataloader]               target dataset'''
        sample_buffer = []
        target_buffer = []
        score_buffer = []
        correct_label = 0
        total = 0
        pseudo_size = 0
        tgt_dataloader = utils.get_data_loader(dataset, self.batch_size, 
                                                cuda=True if self.device=='cuda' else False)
        with torch.no_grad():
            self.model.classifier.eval()
            for images, labels in tgt_dataloader:
                total += len(labels)

                images = images.to(self.device)
                y_hat = self.model.classifier(images)
                if active_classes is not None:
                    class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                    y_hat = y_hat[:, class_entries]

                # Calculate prediction loss
                predict_out = torch.nn.Softmax(dim=1)(y_hat)
                confidence, predict = torch.max(predict_out, 1)
                if active_classes is not None:
                    predict = predict+class_entries[0]

                # keep pseudo samples
                for i, (x,y,y_) in enumerate(zip(images,labels,predict)):
                    if confidence[i].item() > self.pseudo_threshold:
                        y_ps = y_.long() # TODO:return correct infer class
                        if x.shape[0] == 3:
                            x = x.unsqueeze(0)
                        sample_buffer.append(x)
                        target_buffer.append(y_ps)
                        score_buffer.append(confidence[i].item())

                        pseudo_size += 1
                        if y_ps.cpu() == y:
                            correct_label += 1

        self.model.classifier.train()
        assert len(sample_buffer)>0, 'No target pseudo samples generated! May change the threshold.'
        trn_data = {'x': [], 'y': []}
        trn_data['x'] = torch.cat(sample_buffer).cpu()
        if trn_data['x'].shape[-1] == 28:
            trn_data['x'] = trn_data['x'].unsqueeze(1)
        trn_data['y'] = torch.stack(target_buffer).cpu().long()

        # save to pseudo dataset examplars
        if self.exemplars_dataset2._is_active():
            pseudo_cls = np.unique(trn_data['y'],return_counts=True)[0]
            cls_counts = np.unique(trn_data['y'],return_counts=True)[1]
            self.exemplars_dataset2._exemplars_per_class_num(self.active_classes)
            # self.exemplars_dataset2.max_num_exemplars_per_class = self.exemplars_dataset2.max_num_exemplars_per_class \
            #     if self.exemplars_dataset2.max_num_exemplars_per_class<cls_counts.max() else cls_counts.max()
            pseudo_index = []
            for c in pseudo_cls:
                idx = np.argsort(np.array(score_buffer)[np.where(trn_data['y']==c)[0]])[::-1][:self.exemplars_dataset2.max_num_exemplars_per_class]
                idx = np.where(trn_data['y']==c)[0][idx]
                [pseudo_index.append(i) for i in idx]
            
            replay_data = {'x': [], 'y': []}
            replay_data['x'] = trn_data['x'][pseudo_index]
            replay_data['y'] = trn_data['y'][pseudo_index]
            # cls_counts_ = np.unique(replay_data['y'],return_counts=True)[1]
            # print('Exampler target:',cls_counts, '-->', cls_counts_)
            self.pseudo_samples = TensorDataset(replay_data['x'],replay_data['y'])
            
        transformed_dataset = TensorDataset(trn_data['x'],trn_data['y'])

        # test
        verbose=False
        if verbose:
            print("Correct target label: {:5.2f}% ({} out of {} from total {})".format((correct_label/pseudo_size)*100,correct_label,pseudo_size, total))

        return transformed_dataset

    def generate_pseudo_labels_by_top(self, dataset, active_classes=None):
        '''Generate target pseudo samples

        [tgt_dataloader]               target dataset'''
        thresfold = self.pseudo_threshold
        for iter in range(100):
            is_empty=True
            sample_buffer = []
            target_buffer = []
            score_buffer = []
            correct_label = 0
            total = 0
            pseudo_size = 0
            tgt_dataloader = utils.get_data_loader(dataset, self.batch_size, 
                                                    cuda=True if self.device=='cuda' else False)
            with torch.no_grad():
                self.model.classifier.eval()
                for images, labels in tgt_dataloader:
                    total += len(labels)

                    images = images.to(self.device)
                    y_hat = self.model.classifier(images)
                    if active_classes is not None:
                        class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                        y_hat = y_hat[:, class_entries]

                    # Calculate prediction loss
                    predict_out = torch.nn.Softmax(dim=1)(y_hat)
                    confidence, predict = torch.max(predict_out, 1)
                    if active_classes is not None:
                        predict = predict+class_entries[0]

                    # keep pseudo samples
                    for i, (x,y,y_) in enumerate(zip(images,labels,predict)):
                        if confidence[i].item() > thresfold:
                            y_ps = y_.long() # TODO:return correct infer class
                            if x.shape[0] == 3:
                                x = x.unsqueeze(0)
                            sample_buffer.append(x)
                            target_buffer.append(y_ps)
                            score_buffer.append(confidence[i].item())

                            pseudo_size += 1
                            if y_ps.cpu() == y:
                                correct_label += 1

                    if(is_empty):
                        predictions = predict.cpu()
                        scores = confidence.cpu()
                        samples = images.cpu()
                        targets = labels.cpu()
                        is_empty = False
                    else:
                        predictions = torch.cat((predictions, predict.cpu()),dim=0)
                        scores = torch.cat((scores, confidence.cpu()),dim=0)
                        samples = torch.cat((samples, images.cpu()),dim=0)
                        targets = torch.cat((targets, labels.cpu()),dim=0)
        
            if len(target_buffer)<30:
                thresfold = thresfold-0.1
            else:
                self.thresholds.append(thresfold)
                break

        self.model.classifier.train()
        if len(sample_buffer)<(self.exemplars_dataset2.max_num_exemplars_per_class)*len(active_classes):
            pseudo_cls,cls_counts = np.unique(predictions,return_counts=True)
            pseudo_index = []
            for c in pseudo_cls:
                idx = np.argsort(np.array(scores)[np.where(predictions==c)[0]])[::-1][:self.exemplars_dataset2.max_num_exemplars_per_class]
                idx = np.where(predictions==c)[0][idx]
                [pseudo_index.append(i) for i in idx]
            sample_buffer = samples[pseudo_index]
            target_buffer = predictions[pseudo_index]
            score_buffer = scores[pseudo_index]
            trn_data = {'x': [], 'y': []}
            trn_data['x'] = sample_buffer
            trn_data['y'] = target_buffer.long()
            correct_label = sum(target_buffer == targets[pseudo_index]).item()
            pseudo_size = len(trn_data['y'])
            total = len(targets)
        else:
            del predictions,scores,samples,targets
            trn_data = {'x': [], 'y': []}
            trn_data['x'] = torch.cat(sample_buffer).cpu()
            if trn_data['x'].shape[-1] == 28:
                trn_data['x'] = trn_data['x'].unsqueeze(1)
            trn_data['y'] = torch.stack(target_buffer).cpu().long()

        # save to pseudo dataset examplars
        if self.exemplars_dataset2._is_active():
            pseudo_cls = np.unique(trn_data['y'],return_counts=True)[0]
            cls_counts = np.unique(trn_data['y'],return_counts=True)[1]
            self.exemplars_dataset2._exemplars_per_class_num(self.active_classes)
            # self.exemplars_dataset2.max_num_exemplars_per_class = self.exemplars_dataset2.max_num_exemplars_per_class \
            #     if self.exemplars_dataset2.max_num_exemplars_per_class<cls_counts.max() else cls_counts.max()
            pseudo_index = []
            for c in pseudo_cls:
                idx = np.argsort(np.array(score_buffer)[np.where(trn_data['y']==c)[0]])[::-1][:self.exemplars_dataset2.max_num_exemplars_per_class]
                idx = np.where(trn_data['y']==c)[0][idx]
                [pseudo_index.append(i) for i in idx]
            
            replay_data = {'x': [], 'y': []}
            replay_data['x'] = trn_data['x'][pseudo_index]
            replay_data['y'] = trn_data['y'][pseudo_index]
            cls_counts_ = np.unique(replay_data['y'],return_counts=True)[1]
            print('Exampler target:',cls_counts, '-->', cls_counts_)
            self.pseudo_samples = TensorDataset(replay_data['x'],replay_data['y'])
            
        transformed_dataset = TensorDataset(trn_data['x'],trn_data['y'])

        # test
        verbose=True
        if verbose:
            print("Correct target label: {:5.2f}% ({} out of {} from total {})".format((correct_label/pseudo_size)*100,correct_label,pseudo_size, total))
            cls_counts = np.unique(trn_data['y'],return_counts=True)
            print('Exampler target:',cls_counts)

        return transformed_dataset

    def _get_memory_samples(self, dataset):
        is_empty=True
        dataloader = utils.get_data_loader(dataset, self.batch_size, 
                                                cuda=True if self.device=='cuda' else False)
        with torch.no_grad():
            for images, labels in dataloader:
                if(is_empty):
                    samples = images.cpu()
                    targets = labels.cpu()
                    is_empty = False
                else:
                    samples = torch.cat((samples, images.cpu()),dim=0)
                    targets = torch.cat((targets, labels.cpu()),dim=0)
        return samples,targets
        
    def train_loop(self, t, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader):
        """Contains the epochs loop"""

        self.iter_counts = 0
        # train source
        super().train_loop(t, trn_src_loader, val_src_loader, trn_tgt_loader, val_tgt_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset1.collect_exemplars(self.active_classes[-self.classes_per_task:], deepcopy(trn_src_loader))
        if hasattr(self, "pseudo_samples"):
            self.exemplars_dataset2.collect_exemplars(self.active_classes[-self.classes_per_task:], self.pseudo_samples)
            # print('Exemplar 2 size is', len(self.exemplars_dataset2))

    def train_epoch(self, t, e, trn_src_loader, val_src, trn_tgt, val_tgt):
        'iterations'
        trn_src = deepcopy(trn_src_loader)
        discriminator_criterion = torch.nn.CrossEntropyLoss()
        # switch to train mode
        self.model.classifier.train()
        self.model.domain_adv.train()
        # initial w
        c_lr_s = [1, 1, 1]
        c_lr_t = [1, 1, 1]
        ###########################
        ### prepare data loader ###
        ###########################
        # add exemplars to source set
        if len(self.exemplars_dataset1) > 0 and t > 0:
            # source memory set
            # mem_src_loader = utils.get_data_loader(self.exemplars_dataset1, batch_size=self.batch_size, cuda=self.device, 
            #                 drop_last=True if len(self.exemplars_dataset1)>self.batch_size else False)
            # mem_src_loader = ForeverDataIterator(mem_src_loader)
            mem_x_s,mem_y_s = self._get_memory_samples(deepcopy(self.exemplars_dataset1))
            trans_mem_x_s, trans_mem_y_s = self.generate_transformed_samples(mem_x_s,mem_y_s)
            p_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
            p_dir = os.path.join(p_dir, 'images')
            save_image(mem_x_s, os.path.join(p_dir, 'mem_src_task{}.jpg'.format(t+1)))
            save_image(trans_mem_x_s, os.path.join(p_dir, '3xtrf_mem_src_task{}.jpg'.format(t+1)))
            if hasattr(self.exemplars_dataset1.dataset,"sub_indeces"):
                # trn_src_cur = copy.deepcopy(trn_src)
                trn_src.cur_task_indeces = trn_src.sub_indeces
                trn_src.sub_indeces = trn_src.sub_indeces + self.exemplars_dataset1.dataset.sub_indeces
                # print(np.unique(trn_src.targets,return_counts=True))
            elif hasattr(self.exemplars_dataset1.dataset,"index"):
                # trn_src_cur = copy.deepcopy(trn_src)
                trn_src.cur_task_indeces = trn_src.index
                trn_src.index = trn_src.index + self.exemplars_dataset1.dataset.index
                trn_src.samples = np.concatenate((trn_src.samples,self.exemplars_dataset1.dataset.samples),0)
                trn_src.targets = np.concatenate((trn_src.targets, self.exemplars_dataset1.dataset.targets),0)
        trn_src_loader = utils.get_data_loader(trn_src, 
                                self.batch_d, cuda=self.device, drop_last=True)
        trn_tgt_loader = utils.get_data_loader(trn_tgt, 
                                self.batch_d, cuda=self.device, drop_last=True)
        trn_src_loader = ForeverDataIterator(trn_src_loader)
        trn_tgt_loader = ForeverDataIterator(trn_tgt_loader)
        # # source validation set
        # if self.meta:
        #     src_va_dataset = self.generate_transformed_dataset(t, trn_src,balance_data=True)
        #     src_va_loader = utils.get_data_loader(src_va_dataset, batch_size=self.batch_size, cuda=self.device, 
        #                             drop_last=True if len(src_va_dataset)>self.batch_size else False)
        #     src_va_loader = ForeverDataIterator(src_va_loader)

        phase = 'domain invariant'
        # for param in self.model.classifier.backbone.parameters():
        #     param.requires_grad = True
        if e >= self.warmup_epochs:
            phase = 'adaptive assessment'
            # for param in self.model.classifier.backbone.parameters():
            #     param.requires_grad = False
            # pseudo target set
            if self.pseudo:
                if e==self.warmup_epochs:
                    self.pseudo_training_dataset = self.generate_pseudo_labels_by_top(deepcopy(trn_tgt), self.active_classes[-self.classes_per_task:])
                    # pseudo_training_dataset = self.generate_pseudo_labels_by_threshold(trn_tgt, self.active_classes[-self.classes_per_task:])
                pseudo_training_dataset = self.pseudo_training_dataset
                if len(self.exemplars_dataset2) > 0 and t > 0:
                    if hasattr(pseudo_training_dataset,"images"):
                        pseudo_training_dataset.images = np.concatenate((pseudo_training_dataset.images,self.exemplars_dataset2.dataset.images),0)
                        pseudo_training_dataset.labels = torch.cat((pseudo_training_dataset.labels,self.exemplars_dataset2.dataset.labels),0)
                    elif hasattr(pseudo_training_dataset,"tensors"):
                        pseudo_training_dataset.tensors = (
                            torch.cat((pseudo_training_dataset.tensors[0],self.exemplars_dataset2.dataset.tensors[0]),0),
                            torch.cat((pseudo_training_dataset.tensors[1],self.exemplars_dataset2.dataset.tensors[1]),0)
                        )
                    # mem_tgt_loader = utils.get_data_loader(self.exemplars_dataset2, batch_size=self.batch_size, cuda=self.device, 
                    #                 drop_last=True if len(self.exemplars_dataset2)>self.batch_size else False)
                    # mem_tgt_loader = ForeverDataIterator(mem_tgt_loader)
                    mem_x_t,mem_y_t = self._get_memory_samples(deepcopy(self.exemplars_dataset2))
                    trans_mem_x_t, trans_mem_y_t = self.generate_transformed_samples(mem_x_t,mem_y_t)
                    p_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
                    p_dir = os.path.join(p_dir, 'images')
                    save_image(mem_x_t, os.path.join(p_dir, 'mem_tgt_task{}.jpg'.format(t+1)))
                    save_image(trans_mem_x_t, os.path.join(p_dir, '3xtrf_mem_tgt_task{}.jpg'.format(t+1)))
                # print(np.unique(pseudo_training_dataset.tensors[1],return_counts=True))
                pseudo_loader = utils.get_data_loader(pseudo_training_dataset, self.batch_size, cuda=self.device, 
                                        drop_last=True if (len(pseudo_training_dataset)>self.batch_size) \
                                            and (len(pseudo_training_dataset)%self.batch_size<int(self.batch_size/2)) else False)
                pseudo_loader = ForeverDataIterator(pseudo_loader)
            # target validation set -- domain confusion samples from source domain
            if self.pseudo and self.meta:
                # tgt_va_dataset = self.generate_val_tgt(t, trn_src, multiple_transform=True)
                # tgt_va_loader = utils.get_data_loader(tgt_va_dataset, batch_size=self.batch_size, cuda=self.device, 
                #                         drop_last=True if len(tgt_va_dataset)>self.batch_size else False)
                # tgt_va_loader = ForeverDataIterator(tgt_va_loader)
                x_fu, y_fu = self.generate_val_tgt(t, trn_src, multiple_transform=True)

        # iterations
        iters_per_epoch = max(len(trn_src_loader),len(trn_tgt_loader)) \
            if (e < self.pre_epochs) and self.domain_invariant else len(trn_src_loader)
        # print("Source dataloader batch no. {} \t Target dataloader batch no. {}\n\tIterations per epoch is {}"
        #         .format(len(trn_src_loader),len(trn_tgt_loader),iters_per_epoch))

        for i in range(iters_per_epoch):
            
            self.iter_counts += 1
            x_s, labels_s = next(trn_src_loader)
            x_t, _ = next(trn_tgt_loader)
            # if len(self.exemplars_dataset1) > 0 and (t > 0):
            #     x_mem, y_mem = next(mem_src_loader)

            if phase == 'adaptive assessment':
                # x_trans_mem = y_trans_mem = None
                if len(self.exemplars_dataset1) > 0 and (t > 0):
                    if self.meta:
                        # x_s_va, y_s_va = next(src_va_loader)
                        x_s_va, y_s_va = self.generate_transformed_samples(x_s, labels_s)
                        # x_trans_mem, y_trans_mem = self.generate_transformed_samples(x_mem, y_mem) # TODO imb-dataset down sampling?
                        # print("[Inner Loop] Train Assessor")
                        for m in range(self.meta_updates):
                            # self.train_one_step_assessor(t, x_s_va, y_s_va, x_trans_mem, y_trans_mem, self.model.assessorS, self.optimizer1)
                            self.train_one_step_assessor(t, x_s_va, y_s_va, trans_mem_x_s, trans_mem_y_s, self.model.assessorS, self.optimizer1)
                            # print("[Outer Loop] Train Base ")
                            with torch.no_grad():
                                c_lr_s = self.model.assessorS(x_s.to(self.device))
                                c_lr_s = c_lr_s.cpu().detach()
                            # print(c_lr_s)
            
            # compute output
            # x = torch.cat((x_s, x_t), dim=0)
            # y, f = self.model.classifier(x.to(self.device))
            # y_s, _ = y.chunk(2, dim=0)
            y_s, f_s = self.model.classifier(x_s.to(self.device))
            # -if needed, remove predictions for classes not in current task
            if self.active_classes is not None:
                class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                y_s = y_s[:, class_entries]
            cls_loss_s = self.cecriterion(t, y_s, labels_s.to(self.device))

            if len(self.exemplars_dataset1) > 0 and (t > 0) and (phase == 'adaptive assessment'):
                for o in range(self.base_updates):
                    curr_mem_outputs_s, _ = self.model.classifier(mem_x_s.to(self.device))
                    mem_outputs_s_old  = self.model_old(mem_x_s.to(self.device))
                    if self.active_classes is not None:
                        class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                        curr_mem_outputs_s = curr_mem_outputs_s[:, class_entries]
                        mem_outputs_s_old = mem_outputs_s_old[:, class_entries]
                    derloss = self.dercriterion(t, mem_outputs_s_old, curr_mem_outputs_s, \
                        mem_y_s.to(self.device)) if self.der else 0
                    dloss = self.distcriterion(t, curr_mem_outputs_s, mem_outputs_s_old) if self.distill else 0
                    loss = c_lr_s[0] * cls_loss_s + (c_lr_s[1] * derloss * t + c_lr_s[2] * dloss * t) * self.alpha
            else:
                loss = c_lr_s[0] * cls_loss_s

            self.logger.log_scalar(task=t, iter=i + 1, name="loss src", value=loss.item(), group="train-epoch")
            
            if (phase == 'domain invariant') and self.domain_invariant:                
                p = float(i + e * iters_per_epoch) / self.warmup_epochs / iters_per_epoch
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                domain_label = torch.zeros(len(x_s))
                domain_label = domain_label.long()
                output_domain_s = self.model.domain_adv(f_s, alpha)
                transfer_loss_s = discriminator_criterion(output_domain_s, domain_label.to(self.device))
                
                domain_label = torch.ones(len(x_t))
                domain_label = domain_label.long()
                _, f_t = self.model.classifier(x_t.to(self.device))
                output_domain_t = self.model.domain_adv(f_t, alpha)
                transfer_loss_t = discriminator_criterion(output_domain_t, domain_label.to(self.device))

                loss = loss + (transfer_loss_s + transfer_loss_t) * self.trade_off
                
                self.logger.log_scalar(task=t, iter=i + 1, name="domain loss source", value=(transfer_loss_s * self.trade_off).item(), group="train-epoch")
                self.logger.log_scalar(task=t, iter=i + 1, name="domain loss target", value=(transfer_loss_t * self.trade_off).item(), group="train-epoch")
            
            if (phase == 'adaptive assessment') and self.pseudo:
                x_p, labels_p = next(pseudo_loader)
                x_trans_mem = y_trans_mem = None
                if len(self.exemplars_dataset2) > 0 and (t > 0):
                    # x_mem, y_mem = next(mem_tgt_loader)
                    if self.meta:
                        # x_fu, y_fu = next(tgt_va_loader)
                        # x_trans_mem, y_trans_mem = self.generate_transformed_samples(x_mem, y_mem)
                        # print("[Inner Loop] Train Assessor")
                        for m in range(self.meta_updates):
                            # self.train_one_step_assessor(t, x_fu, y_fu, x_trans_mem, y_trans_mem, self.model.assessorT, self.optimizer2)
                            self.train_one_step_assessor(t, x_fu, y_fu, trans_mem_x_t, trans_mem_y_t, self.model.assessorT, self.optimizer2)
                            # print("[Outer Loop] Train Base ")
                            with torch.no_grad():
                                c_lr_t = self.model.assessorT(x_p.to(self.device))
                                c_lr_t = c_lr_t.cpu().detach()
                        # print(c_lr_t)
            
                # compute output
                y_p, _ = self.model.classifier(x_p.to(self.device))
                # -if needed, remove predictions for classes not in current task
                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    y_p = y_p[:, class_entries]
                cls_loss_p = self.cecriterion(t, y_p, labels_p.to(self.device))

                if len(self.exemplars_dataset2) > 0 and (t > 0):
                    for o in range(self.base_updates):
                        curr_mem_outputs_t, _ = self.model.classifier(mem_x_t.to(self.device))
                        mem_outputs_t_old  = self.model_old(mem_x_t.to(self.device))
                        if self.active_classes is not None:
                            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                            curr_mem_outputs_t = curr_mem_outputs_t[:, class_entries]
                            mem_outputs_t_old = mem_outputs_t_old[:, class_entries]
                        derloss = self.dercriterion(t, mem_outputs_t_old, curr_mem_outputs_t, 
                                                    mem_y_t.to(self.device))  if self.der else 0
                        dloss = self.distcriterion(t, curr_mem_outputs_t, 
                                                    mem_outputs_t_old) if self.distill else 0
                        loss_pse = c_lr_t[0] * cls_loss_p + (c_lr_t[1] * derloss * t + c_lr_t[2] * dloss * t)
                        loss = loss + loss_pse * self.alpha
                else:
                    loss_pse = c_lr_t[0] * cls_loss_p
                    loss = loss + loss_pse
                
                self.logger.log_scalar(task=t, iter=i + 1, name="loss pseudo", value=loss_pse.item(), group="train-epoch")
            
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), self.clipgrad)
            torch.nn.utils.clip_grad_norm_(self.model.domain_adv.parameters(), self.clipgrad)
            self.optimizer.step()
            
            self.logger.log_scalar(task=t, iter=i + 1, name="loss", value=loss.item(), group="train-epoch")
            # print('general loss:',loss.item())

    def train_one_step_assessor(self, t, x, y, x_mem: None, y_mem: None, assessor, assessorOptimizer):
        # torch.cuda.empty_cache()
        self.model.classifier.train()
        assessor.train()
        
        outputs, _ = self.model.classifier(x.to(self.device))
        w = assessor(x.to(self.device))
        # print(w)         
        if self.active_classes is not None:
            class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
            outputs = outputs[:, class_entries]      
        
        if(t>0) and x_mem is not None:
            trans_mem_outputs_old = self.model_old(x_mem.to(self.device))
            torch.cuda.empty_cache()
            curr_trans_mem_outputs, _ = self.model.classifier(x_mem.to(self.device))
            if self.active_classes is not None:
                class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                trans_mem_outputs_old = trans_mem_outputs_old[:, class_entries]
                curr_trans_mem_outputs = curr_trans_mem_outputs[:, class_entries]
                
            loss1 = self.cecriterion(t, outputs, y.to(self.device).type(torch.long))
            loss2 = self.dercriterion(t, trans_mem_outputs_old, curr_trans_mem_outputs, y_mem) if self.der else 0
            dloss = self.distcriterion(t, curr_trans_mem_outputs, trans_mem_outputs_old) if self.distill else 0
            loss = w[0]*loss1+(w[1]*loss2*t+w[2]*dloss*t)*self.alpha
        else:
            loss = w[0]*self.cecriterion(t, outputs, y.to(self.device).type(torch.long))

        assessorOptimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(assessor.parameters(), self.clipgrad)
        assessorOptimizer.step()
        # print('Asse:', loss.item(),end=' ')

        assessor.eval()
        torch.cuda.empty_cache()
        self.model.classifier.train()

    def train_assessor(self, t, T_vl, x_mem:None, y_mem:None, assessor, assessorOptimizer):
        torch.cuda.empty_cache()
        dataloader = utils.get_data_loader(T_vl, self.batch_size, cuda=self.device, drop_last=True)
        self.model.classifier.train()
        assessor.train()
        
        for x,y in dataloader:
            outputs, _ = self.model.classifier(x.to(self.device))
            w = assessor(x.to(self.device))
            # print(w)         
            if self.active_classes is not None:
                class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                outputs = outputs[:, class_entries]      
            
            if(t>0) and x_mem is not None:
                trans_mem_outputs_old = self.model_old(x_mem.to(self.device))
                torch.cuda.empty_cache()
                curr_trans_mem_outputs, _ = self.model.classifier(x_mem.to(self.device))
                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    trans_mem_outputs_old = trans_mem_outputs_old[:, class_entries]
                    curr_trans_mem_outputs = curr_trans_mem_outputs[:, class_entries]
                    
                loss1 = self.cecriterion(t, outputs, y.to(self.device).type(torch.long))
                loss2 = self.dercriterion(t, trans_mem_outputs_old, curr_trans_mem_outputs, y_mem) if self.der else 0
                dloss = self.distcriterion(t, curr_trans_mem_outputs, trans_mem_outputs_old) if self.distill else 0
                loss = w[0]*loss1+(w[1]*loss2*t+w[2]*dloss*t)*self.alpha
            else:
                loss = w[0]*self.cecriterion(t, outputs, y.to(self.device).type(torch.long))

            assessorOptimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(assessor.parameters(), self.clipgrad)
            assessorOptimizer.step()
            # print('assessor loss:',loss.item())

        assessor.eval()
        torch.cuda.empty_cache()
        self.model.classifier.train()

    def train_assessor(self, t, T_vl, T_mem:None, assessor, assessorOptimizer):
        torch.cuda.empty_cache()
        dataloader = utils.get_data_loader(T_vl, self.batch_size, cuda=self.device, drop_last=True)
        if T_mem is not None:
            mem = utils.get_data_loader(T_mem, self.batch_size, cuda=self.device, drop_last=True)
            mem = ForeverDataIterator(mem)
        self.model.classifier.train()
        assessor.train()
        
        for x,y in dataloader:
            outputs, _ = self.model.classifier(x.to(self.device))
            w = assessor(x.to(self.device))
            # print(w)         
            if self.active_classes is not None:
                class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                outputs = outputs[:, class_entries]      
            
            if(t>0) and T_mem is not None:
                x_mem,y_mem=next(mem)
                trans_mem_outputs_old = self.model_old(x_mem.to(self.device))
                torch.cuda.empty_cache()
                curr_trans_mem_outputs, _ = self.model.classifier(x_mem.to(self.device))
                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    trans_mem_outputs_old = trans_mem_outputs_old[:, class_entries]
                    curr_trans_mem_outputs = curr_trans_mem_outputs[:, class_entries]
                    
                loss1 = self.cecriterion(t, outputs, y.to(self.device).type(torch.long))
                loss2 = self.dercriterion(t, trans_mem_outputs_old, curr_trans_mem_outputs, y_mem) if self.der else 0
                dloss = self.distcriterion(t, curr_trans_mem_outputs, trans_mem_outputs_old) if self.distill else 0
                loss = w[0]*loss1+(w[1]*loss2*t+w[2]*dloss*t)*self.alpha
            else:
                loss = w[0]*self.cecriterion(t, outputs, y.to(self.device).type(torch.long))

            assessorOptimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(assessor.parameters(), self.clipgrad)
            assessorOptimizer.step()
            # print('assessor loss:',loss.item())

        assessor.eval()
        torch.cuda.empty_cache()
        self.model.classifier.train()

    # def pre_train_process(self, t, trn_src_loader, trn_tgt_loader):
    #     """Runs before training all epochs of the task (before the train session)"""
    #     # Warm-up phase
    #     if self.warmup_epochs:
    #         finetune = True
    #         params = [
    #             {"params": self.model.classifier.backbone.parameters(), "lr": 0.1 * self.lr1 if finetune else 1.0 * self.lr1},
    #             {"params": self.model.classifier.bottleneck.parameters(), "lr": 1.0 * self.lr1},
    #             {"params": self.model.classifier.head.parameters(), "lr": 1.0 * self.lr1},
    #             {"params": self.model.domain_adv.domain_discriminator.parameters(), "lr":  1.0 * self.lr1}]
    #         torch.optim.SGD(params, lr=self.lr1, weight_decay=self.wd, momentum=self.momentum)
    #         self.optimizer = torch.optim.SGD(params, lr=self.lr1, weight_decay=self.wd, momentum=self.momentum)
    #         # Loop epochs -- train warm-up head
    #         for e in range(self.warmup_epochs):
    #             trn_src_loader = utils.get_data_loader(trn_src_loader,self.batch_d,cuda=True if self.device=='cuda' else False, drop_last=True)
    #             trn_tgt_loader = utils.get_data_loader(trn_tgt_loader,self.batch_d,cuda=True if self.device=='cuda' else False, drop_last=True)
    #             self.model.classifier.train()
    #             warmupclock0 = time.time()
    #             for (x_s, labels_s), (x_t, _) in zip(trn_src_loader, trn_tgt_loader):
    #                 x_s = x_s.to(self.device)
    #                 labels_s = labels_s.to(self.device)
    #                 x_t = x_t.to(self.device)

    #                 # lr_scheduler.step()

    #                 # compute output
    #                 x = torch.cat((x_s, x_t), dim=0)
    #                 y, f = self.model.classifier(x)
    #                 y_s, y_t = y.chunk(2, dim=0)
    #                 f_s, f_t = f.chunk(2, dim=0)

    #                 # -if needed, remove predictions for classes not in current task
    #                 if self.active_classes is not None:
    #                     class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
    #                     y_s = y_s[:, class_entries]
    #                 cls_loss = self.cecriterion(t, y_s, labels_s)
                    
    #                 transfer_loss = self.model.domain_adv(f_s, f_t)
    #                 # domain_acc = self.model.domain_adv.domain_discriminator_accuracy
                    
    #                 loss = cls_loss + transfer_loss * self.trade_off

    #                 # compute gradient and do SGD step
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), self.clipgrad)
    #                 self.optimizer.step()

    #             warmupclock1 = time.time()
    #             with torch.no_grad():
    #                 total_loss, total_acc_taw = 0, 0
    #                 self.model.classifier.eval()
    #                 for images, targets in trn_tgt_loader:
    #                     outputs = self.model.classifier(images.to(self.device))
    #                     loss = self.cecriterion(t, outputs, targets.to(self.device))
    #                     if self.active_classes is not None:
    #                         class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
    #                         outputs = outputs[:, class_entries]
    #                     predict_out = torch.nn.Softmax(dim=1)(outputs)
    #                     confidence, predict = torch.max(predict_out, 1)
    #                     hits_taw = (predict == targets.to(self.device)).float()
    #                     total_loss += loss.item() * len(targets)
    #                     total_acc_taw += hits_taw.sum().item()
    #             total_num = len(trn_tgt_loader.dataset)
    #             trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
    #             warmupclock2 = time.time()
    #             print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
    #                 e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
    #             self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
    #             self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")


    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model.classifier)
        self.model_old.eval()
        self.model_old.freeze_all()

    def eval(self, t, dataset):
        """Contains the evaluation code"""
        val_loader = utils.get_data_loader(dataset, self.batch_size, cuda=True if self.device=='cuda' else False)
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model.classifier(images.to(self.device))
                loss = self.cecriterion(t, outputs, targets.to(self.device).type(torch.long))
                # hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                if self.active_classes is not None:
                    class_entries = self.active_classes[-1] if type(self.active_classes[0])==list else self.active_classes
                    outputs = outputs[:, class_entries]
                # Calculate prediction loss
                predict_out = torch.nn.Softmax(dim=1)(outputs)
                confidence, predict = torch.max(predict_out, 1)
                hits_taw = (predict == targets.to(self.device)).float()
                predict = outputs.argmax(1)
                hits_tag = (predict == targets.to(self.device)).float()
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cecriterion(self, t, outputs, targets):
        targets = targets.long()
        return torch.nn.functional.cross_entropy(outputs, targets)
    
    def distcriterion(self, t, outputs, outputs_old=None):
        g = torch.sigmoid(outputs)
        q_i = torch.sigmoid(outputs_old)
        loss = sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(outputs_old.shape[1])) # sum(self.model.task_cls[:t]))
        return loss

    def dercriterion(self, t, prev_mem_outputs, curr_mem_outputs, mem_targets):
        a1=1.0
        a2=1.0
        # l2 = torch.norm(curr_mem_outputs-prev_mem_outputs,2)
        l2 = F.mse_loss(curr_mem_outputs, prev_mem_outputs)
        return (a1*l2)+(a2*torch.nn.functional.cross_entropy(curr_mem_outputs, mem_targets.type(torch.long).to(self.device)))