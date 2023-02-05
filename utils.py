import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn import functional as F
from torchvision import transforms
import copy
import data
import models
import timm

import matplotlib
matplotlib.use('Agg')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as col

cudnn_deterministic = True

###################
### backbone    ###
###################

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()
    

AVAILABLE_NORMS = {
    'splitMNIST': transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    'splitUSPS' : transforms.Normalize((0.5,), (0.5,)),
    'Office-31':  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'Office-Home':transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
}


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

class ReverseLayerF(torch.autograd.Function):    
    @staticmethod    
    def forward(ctx, x, alpha):        
        ctx.alpha = alpha        
        return x.view_as(x)    
    
    @staticmethod    
    def backward(ctx, grad_output):        
        output = grad_output.neg() * ctx.alpha        
        return output, None

###################
## Loss function ##
###################

def loss_fn_kd(scores, target_scores, T=2.):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2.):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss
    KD_loss_unnorm = -( targets_norm * torch.log(scores_norm) + (1-targets_norm) * torch.log(1-scores_norm) )
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)      #--> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()          #--> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T**2

    return KD_loss


##-------------------------------------------------------------------------------------------------------------------##


#############################
## Data-handling functions ##
#############################

def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False, shuffle=True):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle=shuffle,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        # **({'num_workers': 4, 'pin_memory': True} if cuda else {})
    )


def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()


def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c


##-------------------------------------------------------------------------------------------------------------------##

##########################################
## Object-saving and -loading functions ##
##########################################

def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

##-------------------------------------------------------------------------------------------------------------------##

################################
## Model-inspection functions ##
################################

def count_parameters(model, verbose=True):
    '''Count number of parameters, print to screen.'''
    total_params = learnable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            learnable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(learnable_params,
                                                                     round(learnable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, learnable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("\n" + 40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-")



##-------------------------------------------------------------------------------------------------------------------##

#################################
## Custom-written "nn-Modules" ##
#################################


class Identity(nn.Module):
    '''A nn-module to simply pass on the input data.'''
    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Reshape(nn.Module):
    '''A nn-module to reshape a tensor to a 4-dim "image"-tensor with [image_channels] channels.'''
    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        image_size = int(np.sqrt(x.nelement() / (batch_size*self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(channels = {})'.format(self.image_channels)
        return tmpstr


class ToImage(nn.Module):
    '''Reshape input units to image with pixel-values between 0 and 1.

    Input:  [batch_size] x [in_units] tensor
    Output: [batch_size] x [image_channels] x [image_size] x [image_size] tensor'''

    def __init__(self, image_channels=1):
        super().__init__()
        # reshape to 4D-tensor
        self.reshape = Reshape(image_channels=image_channels)
        # put through sigmoid-nonlinearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def image_size(self, in_units):
        '''Given the number of units fed in, return the size of the target image.'''
        image_size = np.sqrt(in_units/self.image_channels)
        return image_size


class Flatten(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''
    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


##-------------------------------------------------------------------------------------------------------------------##

def AddGaussianNoise(input, mean=0, std=0.001):
    for i,x in enumerate(input):
        x = x + torch.randn(x.size()) * std + mean
        input[i] = x
    return input

def rotate(input, angle):
    rot = transforms.CenterCrop((28,28))
    input = rot(input)
    return input

##-------------------------------------------------------------------------------------------------------------------##

def get_dataset_samples(dataset):
    samples = []
    targets = []
    for x,y in dataset:
        samples.append(x)
        targets.append(y)
    return torch.from_numpy(np.stack(samples)),torch.from_numpy(np.stack(targets))



##-------------------------------------------------------------------------------------------------------------------##


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.train()
    all_features = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            _, feature = feature_extractor(inputs)
            all_features.append(feature)
    return torch.cat(all_features, dim=0)

def collect_dann_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return
    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            feature = feature_extractor(inputs)
            all_features.append(feature)
    return torch.cat(all_features, dim=0)

def visualize(source_datasets, target_datasets, net:nn.Module, total_task:int, 
              filename:str, colors:list, batch_size:int,
              device='cpu', m_type=1):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
        m_type : 1--clamp (proposed idea) 2--dann
    """
    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    task_feat1 = []
    task_feat2 = []
    # Loop over all tasks.
    for task in range(total_task):  ### Task index starts from 0
        train_source_loader = DataLoader(
            source_datasets[task], batch_size=batch_size, shuffle=False, drop_last=False
        )
        train_target_loader = DataLoader(
            target_datasets[task], batch_size=batch_size, shuffle=False, drop_last=False
        )
        if m_type==1:
            source_feature = collect_feature(train_source_loader, net, device).cpu()
            target_feature = collect_feature(train_target_loader, net, device).cpu()
        elif m_type==2:
            source_feature = collect_dann_feature(train_source_loader, net, device).cpu()
            target_feature = collect_dann_feature(train_target_loader, net, device).cpu()
        
        X_tsne1 = TSNE(n_components=2, random_state=33, perplexity=50, n_iter=1000).fit_transform(source_feature) # map features to 2-d using TSNE
        X_tsne2 = TSNE(n_components=2, random_state=33, perplexity=50, n_iter=1000).fit_transform(target_feature) # map features to 2-d using TSNE
        plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1], c = colors[task], 
                    label="Source--Task {}".format(task)
                    )
        plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c = lighten_color(colors[task]), alpha=0.5,
                    label="Target--Task {}".format(task)
                    )

    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=360)
    print('tSNE plot saved.')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_acc(accs, args, figname, y_label):
    from matplotlib.ticker import FuncFormatter
    iters_task = [len(sum(accs[i*(args.epoch):(i+1)*(args.epoch)],[])) for i in range(args.tasks)]

    second_axis = [0]
    [second_axis.append(sum(iters_task[:i+1])) for i in range(len(iters_task))]

    def perc(x, pos):
        return '{:.0f}%'.format(x*100)

    fig, ax = plt.subplots(constrained_layout=True)
    x = np.arange(len(sum(accs,[])))
    y = sum(accs,[])

    # set the formatter function for the y-axis
    formatter = FuncFormatter(perc)

    ax.plot(x, y)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlim(0, len(sum(accs,[])))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(y_label)
    # ax.set_title(title)

    second_x = ['', 'Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']
    ax2 = ax.twiny()
    plt.grid(color = 'gainsboro', linestyle='--',linewidth = 1)
    ax2.set_xticks(second_axis)
    ax2.set_xticklabels([])
    ax2.set_xticklabels(second_x)
    plt.savefig(figname, dpi=600)