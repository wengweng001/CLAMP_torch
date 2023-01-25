import importlib
from argparse import ArgumentParser

from datasets.memory_dataset import MemoryDataset


# class ExemplarsDataset(MemoryDataset):
#     """Exemplar storage for approaches with an interface of Dataset"""

#     def __init__(self, transform, class_indices,
#                  num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random'):
#         super().__init__({'x': [], 'y': []}, transform, class_indices=class_indices)
#         self.max_num_exemplars_per_class = num_exemplars_per_class
#         self.max_num_exemplars = num_exemplars
#         assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'
#         cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize())
#         selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection'), cls_name)
#         self.exemplars_selector = selector_cls(self)

#     # Returns a parser containing the approach specific parameters
#     @staticmethod
#     def extra_parser(args):
#         parser = ArgumentParser("Exemplars Management Parameters")
#         _group = parser.add_mutually_exclusive_group()
#         _group.add_argument('--num-exemplars', default=0, type=int, required=False,
#                             help='Fixed memory, total number of exemplars (default=%(default)s)')
#         _group.add_argument('--num-exemplars-per-class', default=0, type=int, required=False,
#                             help='Growing memory, number of exemplars per class (default=%(default)s)')
#         parser.add_argument('--exemplar-selection', default='random', type=str,
#                             choices=['herding', 'random', 'entropy', 'distance'],
#                             required=False, help='Exemplar selection strategy (default=%(default)s)')
#         return parser.parse_known_args(args)

#     def _is_active(self):
#         return self.max_num_exemplars_per_class > 0 or self.max_num_exemplars > 0

#     def collect_exemplars(self, model, trn_loader, selection_transform):
#         if self._is_active():
#             self.images, self.labels = self.exemplars_selector(model, trn_loader, selection_transform)

from torch.utils.data import Dataset
import numpy as np
import random
import torch

class ExemplarsDataset(Dataset):
    """Exemplar storage for approaches with an interface of Dataset"""

    def __init__(self,
                 num_exemplars=0, num_exemplars_per_class=0):
        self.dataset = {}
        self.max_num_exemplars_per_class = num_exemplars_per_class
        self.max_num_exemplars = num_exemplars
        assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser("Exemplars Management Parameters")
        _group = parser.add_mutually_exclusive_group()
        _group.add_argument('--num-exemplars', default=0, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        _group.add_argument('--num-exemplars-per-class', default=0, type=int, required=False,
                            help='Growing memory, number of exemplars per class (default=%(default)s)')
        return parser.parse_known_args(args)

    def _is_active(self):
        return self.max_num_exemplars_per_class > 0 or self.max_num_exemplars > 0

    def _exemplars_per_class_num(self, task_cls):
        if self.max_num_exemplars_per_class:
            return self.max_num_exemplars_per_class

        num_cls = len(task_cls)
        num_exemplars = self.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def collect_exemplars(self, task_cls, dataset):
        if self._is_active():
            exemplars_per_class = self._exemplars_per_class_num(task_cls)

            print('Save buffer for cur task of class {}'.format(task_cls))
            index = []
            if hasattr(dataset,"index"):
                if len(self.dataset) > 0:
                    labels = np.concatenate((self.dataset.targets, np.array(dataset.targets)),axis=0)
                    num_cls = np.unique(dataset.targets)
                elif len(self.dataset) == 0:
                    num_cls = np.unique(dataset.targets)
                    labels = np.array(dataset.targets)
            elif hasattr(dataset,"dataset"):
                num_cls = np.unique(np.array(dataset.dataset.targets)[dataset.sub_indeces])
                labels = np.array(dataset.dataset.targets)[dataset.sub_indeces]
            elif hasattr(dataset,"images"):
                if len(self.dataset) == 0:
                    num_cls = np.unique(np.array(dataset.labels))
                    labels = np.array(dataset.labels)
                elif len(self.dataset) > 0:
                    labels = np.array(dataset.labels)
                    labels = np.concatenate((self.dataset.labels,labels),axis=0)
                    num_cls = np.unique(labels)
            elif hasattr(dataset,"tensors"):
                if len(self.dataset) == 0:
                    num_cls = torch.unique(dataset.tensors[1])
                    labels =dataset.tensors[1]
                elif len(self.dataset) > 0:
                    labels = dataset.tensors[1]
                    labels = torch.cat((self.dataset.tensors[1],labels),0)
                    num_cls = torch.unique(labels)
                
                
            if exemplars_per_class>(np.unique(labels,return_counts=True)[1].min()):
                exemplars_per_class = np.round(np.unique(labels,return_counts=True)[1].min(),decimals=-1)
                if exemplars_per_class == 0:
                    exemplars_per_class = np.unique(labels,return_counts=True)[1].min()

            for curr_cls in task_cls:
                # get all indices from current class -- check if there are exemplars from previous task in the loader
                cls_ind = np.where(labels == curr_cls)[0]
                # assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
                # assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
                if (len(cls_ind) == 0):
                    print("No samples to choose from for class {:d}".format(curr_cls))
                    continue
                if (exemplars_per_class > len(cls_ind)):
                    print("Not enough samples to store from for class {:d}".format(curr_cls))
                    index.extend(cls_ind)
                    continue
                # select the exemplars randomly
                index.extend(np.random.choice(cls_ind,exemplars_per_class,replace=False))
            
            if hasattr(dataset,"index"):
                if len(self.dataset) > 0:
                    self.dataset.index = self.dataset.index + np.array(self.dataset.index + dataset.index)[index].tolist()
                    cur_task_x = np.concatenate((self.dataset.samples,dataset.samples),0)[index]
                    self.dataset.samples = np.concatenate((self.dataset.samples,cur_task_x ),0)
                    cur_task_y = np.concatenate((self.dataset.targets,dataset.targets),0)[index]
                    self.dataset.targets = np.concatenate((self.dataset.targets,cur_task_y ),0)
                    return
                self.dataset = dataset
                self.dataset.index = np.array(dataset.index)[index].tolist()
                self.dataset.samples = dataset.samples[index]
                self.dataset.targets = dataset.targets[index]
            elif hasattr(dataset,"dataset"):
                if len(self.dataset) > 0:
                    cur_task_idx = np.array(dataset.sub_indeces)[index].tolist()
                    self.dataset.sub_indeces = self.dataset.sub_indeces + cur_task_idx
                    return
                self.dataset = dataset
                self.dataset.sub_indeces = np.array(dataset.sub_indeces)[index].tolist()
                self.target_transform = self.dataset.dataset.target_transform
                self.transform = self.dataset.dataset.transform
            elif hasattr(dataset,"images"):
                if len(self.dataset) > 0:
                    imgs = np.concatenate((self.dataset.images,dataset.images),axis=0)[index]
                    imgs = np.concatenate((self.dataset.images,imgs),axis=0)
                    labels = torch.cat((self.dataset.labels,dataset.labels),axis=0)[index]
                    labels = torch.cat((self.dataset.labels,labels),axis=0)
                    self.dataset.images = imgs
                    self.dataset.labels = labels
                    return
                self.dataset = dataset
                self.dataset.transform = dataset.transform
                self.dataset.images = dataset.images[sorted(index)]
                self.dataset.labels = dataset.labels[sorted(index)]
            elif hasattr(dataset,"tensors"):
                if len(self.dataset) > 0:
                    x = torch.cat((self.dataset.tensors[0],dataset.tensors[0]),axis=0)[index]
                    x = torch.cat((self.dataset.tensors[0],x),axis=0)
                    y = torch.cat((self.dataset.tensors[1],dataset.tensors[1]),axis=0)[index]
                    y = torch.cat((self.dataset.tensors[1],y),axis=0)
                    self.dataset.tensors = (x,y)
                    return
                self.dataset = dataset
                self.dataset.tensors = (dataset.tensors[0][sorted(index)],dataset.tensors[1][sorted(index)])


    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.dataset)

    def __getitem__(self, index):
        if hasattr(self.dataset,"dataset"):
            sample = self.dataset.dataset[self.dataset.sub_indeces[index]]
            target = torch.tensor(sample[1])
            sample = (sample[0], target)
            if self.target_transform:
                target = self.target_transform(sample[1])
                sample = (sample[0], target)
            return sample
        elif hasattr(self.dataset,"index"):
            path, target = self.dataset.samples[index]
            sample = self.dataset.loader(path)
            if self.dataset.transform is not None:
                sample = self.dataset.transform(sample)
            if self.dataset.target_transform is not None:
                target = self.dataset.target_transform(target)

            return sample, target
        elif hasattr(self.dataset,"images"):
            from PIL import Image
            try:
                x = Image.fromarray(self.dataset.images[index])
            except:
                x = Image.fromarray(self.dataset.images[index], 'RGB')
            x = self.dataset.transform(x)
            y = self.dataset.labels[index]
            return x, y
        elif hasattr(self.dataset,"tensors"):
            return tuple(tensor[index] for tensor in self.dataset.tensors)