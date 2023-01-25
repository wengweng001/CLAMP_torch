from turtle import forward
import torch
from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Dict
from copy import deepcopy
import torchvision

import sys
sys.path.append('./')
import models
import timm

def enum(**enums):
    return type('Enum', (), enums)

DatasetsType = enum(
    office31    = 'Office-31',
    officehome  = 'Office-Home',
    mnist       = 'splitMNIST',
    usps        = 'splitUSPS'
)

def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone

class assessor(nn.Module):
    """Assessor module for CLAMP."""

    def __init__(self, image_size=28, channels=1, h_dim=256):
        """Init assessor."""
        super(assessor, self).__init__()

        self.fc1 = nn.Linear(image_size * image_size * channels, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.relu = nn.ReLU()
        # nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc1.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc2.bias.data.zero_()

        self.lstm= nn.LSTM(input_size=h_dim ,hidden_size=64 ,num_layers=2 ,batch_first=True)
        self.fc3 = nn.Linear(in_features=64 ,out_features=64)
        self.fc4 = nn.Linear(in_features=64 ,out_features=3)
        self.init_param()
        # nn.init.xavier_uniform_(self.lstm.weight)
        # self.lstm.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc3.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc4.weight)
        # self.fc4.bias.data.zero_()


    def init_param(self):
        for name, param in self.named_parameters():
            torch.nn.init.normal_(param);

    def forward(self, x):
        x = x.view(x.size(0) ,-1)
        x = self.fc1(x)
        x = self.fc2(x)

        x = x.view(x.shape[0], 1, x.shape[1])

        h0 = torch.zeros(2, x.shape[0], 64).to(x.device)
        c0 = torch.zeros(2, x.shape[0], 64).to(x.device)

        out ,_ = self.lstm(x ,(h0 ,c0))
        out = out[: ,-1 ,:]
        out = self.fc3(out)
        out = self.fc4(out)
        output = torch.sigmoid(out)
        # output = np.mean(output.cpu().detach().numpy() ,axis=0)
        output = torch.mean(output ,axis=0)
        return output

class LSTMAssessor(nn.Module):

    def __init__(self ,numChannels):

        super(LSTMAssessor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=numChannels ,kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=numChannels, out_channels=8 ,kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8 ,kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.lstm= nn.LSTM(input_size=676 ,hidden_size=128 ,num_layers=2 ,batch_first=True)
        self.fc1 = nn.Linear(in_features=128 ,out_features=64)
        self.fc2 = nn.Linear(in_features=64 ,out_features=3)
        self.init_param()

        # print(self)

    def init_param(self):
        for name, param in self.named_parameters():
            torch.nn.init.normal_(param);

    def forward(self, x):
        # x = torch.flatten(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = x.view(x.shape[0] ,x.shape[1] ,x.shape[2 ] *x.shape[2])

        h0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)

        out ,_ = self.lstm(x ,(h0 ,c0))
        out = out[: ,-1 ,:]
        out = self.fc1(out)
        out = self.fc2(out)
        output = torch.sigmoid(out)
        # output = np.mean(output.cpu().detach().numpy() ,axis=0)
        output = torch.mean(output ,axis=0)
        return output

class assessor_cnn(nn.Module):
    """Assessor module for CLAMP."""

    def __init__(self, h_dim=1024, resnet_34=False, resnet_50=False):
        """Init assessor."""
        super(assessor_cnn, self).__init__()
        from models.resnet import resnet34, resnet50
        if resnet_34:
            self.encoder = resnet34(pretrained=True)
        elif resnet_50:
            self.encoder = resnet50(pretrained=True)
        self.fc = nn.Linear(self.encoder.fc.out_features, h_dim)
        self.relu = nn.ReLU()
        # nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc1.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc2.bias.data.zero_()

        self.lstm= nn.LSTM(input_size=h_dim ,hidden_size=128 ,num_layers=2 ,batch_first=True)
        self.fc1 = nn.Linear(in_features=128 ,out_features=128)
        self.fc2 = nn.Linear(in_features=128 ,out_features=3)
        self.init_param()
        # nn.init.xavier_uniform_(self.lstm.weight)
        # self.lstm.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc3.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc4.weight)
        # self.fc4.bias.data.zero_()


    def init_param(self):
        for name, param in self.named_parameters():
            torch.nn.init.normal_(param);

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        x = x.view(x.shape[0], 1, x.shape[1])

        h0 = torch.zeros(2, x.shape[0], 128).to(self.device)
        c0 = torch.zeros(2, x.shape[0], 128).to(self.device)

        out ,_ = self.lstm(x ,(h0 ,c0))
        out = out[: ,-1 ,:]
        out = self.fc1(out)
        out = self.fc2(out)
        output = torch.sigmoid(out)
        # output = np.mean(output.cpu().detach().numpy() ,axis=0)
        output = torch.mean(output ,axis=0)
        return output

class cnnlstm(nn.Module):
    def __init__(self, backbone, bottleneck, pool, embed_dim, hidden_size, num_layers, num_classes):
        super(cnnlstm, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.backbone = backbone
        self.pool = pool
        self.bottleneck = bottleneck

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=hidden_size),
            # nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        try:
            x = self.pool(self.backbone(x))
        except:
            x = self.backbone(x)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        # lstm part
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        x, (hn, cn) = self.lstm(x, (h_0, c_0))
        x = x[:, -1, :]
        x = self.fc(x)
        output = torch.sigmoid(x)
        x = torch.mean(output ,axis=0)
        return x

class ClassifierBase(nn.Module):
    """A generic Classifier class for domain adaptation.
    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True
    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.
    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.
    Inputs:
        - x (tensor): input data fed to `backbone`
    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer
    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)
    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
            nn.init.normal_(self.head.weight)
            nn.init.constant_(self.head.bias, 0)
        else:
            self.head = head
        self.finetune = finetune
	    
        # self.apply(self._init_weights_xavier)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean = 0, std = 0.01)
            module.bias.data.fill_(0.0)

    def _init_weights_xavier(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        try:
            f = self.pool_layer(self.backbone(x))
        except:
            f = self.backbone(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.
    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                final_layer
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k
    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.
    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
        
from torch.autograd import Function
class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is
    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].
    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.
    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.
    Examples::
        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class FeatrueExtrator(nn.Module):
    """
    
    """
    def __init__(self, backbone: nn.Module,
                 bottleneck_dim: Optional[int] = -1, 
                 pool_layer=None):
        super(FeatrueExtrator, self).__init__()
        self.backbone = backbone
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        assert bottleneck_dim > 0
        self._features_dim = bottleneck_dim

        self.bottleneck.apply(init_weights)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        try:
            f = self.pool_layer(self.backbone(x))
        except:
            f = self.backbone(x)
        x = self.bottleneck(f)
        return x
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class assessor_cnn(nn.Module):
    def __init__(self, backbone, embed_dim, hidden_size, num_layers, num_classes):
        super(assessor_cnn, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.backbone = backbone

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=hidden_size),
            # nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        # lstm part
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        x, (hn, cn) = self.lstm(x, (h_0, c_0))
        x = x[:, -1, :]
        x = self.fc(x)
        output = torch.sigmoid(x)
        x = torch.mean(output ,axis=0)
        return x

class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            # nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


class CLAMP(nn.Module):
    def __init__(self, dataset_name: str, network_name: str, num_classes: int, bottleneck_dim: Optional[int] = 256,
                scratch: bool = True, no_pool: bool = False, pretrained=False):
        super(CLAMP, self).__init__()
        # backbone
        backbone = get_model(network_name, pretrain=pretrained)
        pool_layer = nn.Identity() if no_pool else None
        self.classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=bottleneck_dim,
                                        pool_layer=pool_layer, finetune=not scratch)
        domain_discri = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=bottleneck_dim)
        self.domain_adv = DomainAdversarialLoss(domain_discri)
        
        # if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
        #     self.assessorS = assessor(h_dim=self.classifier.features_dim)
        #     self.assessorT = assessor(h_dim=self.classifier.features_dim)
        # else:
        #     self.assessorS = LSTMAssessor(3)
        #     self.assessorT = LSTMAssessor(3)

        # if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
        #     lstm_nodes = 64
        # else:
        #     lstm_nodes = 128
        # import copy
        # self.assessorS = cnnlstm(backbone=copy.deepcopy(backbone),bottleneck=copy.deepcopy(self.classifier.bottleneck),
        #                          pool=copy.deepcopy(self.classifier.pool_layer), embed_dim=bottleneck_dim,
        #                          hidden_size=lstm_nodes,num_layers=2,num_classes=3)
        # self.assessorT = cnnlstm(backbone=copy.deepcopy(backbone),bottleneck=copy.deepcopy(self.classifier.bottleneck),
        #                          pool=copy.deepcopy(self.classifier.pool_layer), embed_dim=bottleneck_dim,
        #                          hidden_size=lstm_nodes,num_layers=2,num_classes=3)

        if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
            self.assessorS = assessor(h_dim=self.classifier.features_dim)
            self.assessorT = assessor(h_dim=self.classifier.features_dim)
        else:
            lstm_nodes = 128
            import copy
            self.assessorS = cnnlstm(backbone=copy.deepcopy(backbone),bottleneck=copy.deepcopy(self.classifier.bottleneck),
                                    pool=copy.deepcopy(self.classifier.pool_layer), embed_dim=bottleneck_dim,
                                    hidden_size=lstm_nodes,num_layers=2,num_classes=3)
            self.assessorT = cnnlstm(backbone=copy.deepcopy(backbone),bottleneck=copy.deepcopy(self.classifier.bottleneck),
                                    pool=copy.deepcopy(self.classifier.pool_layer), embed_dim=bottleneck_dim,
                                    hidden_size=lstm_nodes,num_layers=2,num_classes=3)

            # lstm_nodes = 128
            # backbone = get_model(network_name, pretrain=pretrained)
            # pool_layer = nn.Identity() if no_pool else None
            # backbone = FeatrueExtrator(backbone, bottleneck_dim, pool_layer)
            # self.assessorS = assessor_cnn(backbone=backbone, embed_dim=bottleneck_dim,
            #                         hidden_size=lstm_nodes, num_layers=2, num_classes=3)
            # self.assessorT = assessor_cnn(backbone=backbone, embed_dim=bottleneck_dim,
            #                         hidden_size=lstm_nodes, num_layers=2, num_classes=3)

    def get_base_parameters(self) -> List[Dict]:
        param = self.classifier.get_parameters() + self.domain_adv.domain_discriminator.get_parameters()
        return param

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return


class CLAMP_new(nn.Module):
    def __init__(self, dataset_name: str, network_name: str, num_classes: int, bottleneck_dim: Optional[int] = 256,
                scratch: bool = True, no_pool: bool = False, pretrained=False):
        super(CLAMP_new, self).__init__()
        # backbone
        backbone = get_model(network_name, pretrain=pretrained)
        pool_layer = nn.Identity() if no_pool else None
        self.classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=bottleneck_dim,
                                        pool_layer=pool_layer, finetune=not scratch)
        domain_discri = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=bottleneck_dim)
        self.domain_adv = DomainAdversarialLoss(domain_discri)
        
        # if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
        #     self.assessorS = assessor(h_dim=self.classifier.features_dim)
        #     self.assessorT = assessor(h_dim=self.classifier.features_dim)
        # else:
        #     self.assessorS = LSTMAssessor(3)
        #     self.assessorT = LSTMAssessor(3)

        # if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
        #     lstm_nodes = 64
        # else:
        #     lstm_nodes = 128
        # import copy
        # self.assessorS = cnnlstm(backbone=copy.deepcopy(backbone),bottleneck=copy.deepcopy(self.classifier.bottleneck),
        #                          pool=copy.deepcopy(self.classifier.pool_layer), embed_dim=bottleneck_dim,
        #                          hidden_size=lstm_nodes,num_layers=2,num_classes=3)
        # self.assessorT = cnnlstm(backbone=copy.deepcopy(backbone),bottleneck=copy.deepcopy(self.classifier.bottleneck),
        #                          pool=copy.deepcopy(self.classifier.pool_layer), embed_dim=bottleneck_dim,
        #                          hidden_size=lstm_nodes,num_layers=2,num_classes=3)

        if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
            self.assessorS = assessor(h_dim=self.classifier.features_dim)
            self.assessorT = assessor(h_dim=self.classifier.features_dim)
        else:
            # lstm_nodes = 256
            lstm_nodes = 128
            # lstm_nodes = 64
            backbone = get_model(network_name, pretrain=pretrained)
            pool_layer = nn.Identity() if no_pool else None
            backbone = FeatrueExtrator(backbone, bottleneck_dim, pool_layer)
            self.assessorS = assessor_cnn(backbone=backbone, embed_dim=bottleneck_dim,
                                    hidden_size=lstm_nodes, num_layers=2, num_classes=3)
            self.assessorT = assessor_cnn(backbone=backbone, embed_dim=bottleneck_dim,
                                    hidden_size=lstm_nodes, num_layers=2, num_classes=3)

    def get_base_parameters(self) -> List[Dict]:
        param = self.classifier.get_parameters() + self.domain_adv.domain_discriminator.get_parameters()
        return param

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

        
class CLAMP_resnet18(nn.Module):
    def __init__(self, dataset_name: str, network_name: str, num_classes: int, bottleneck_dim: Optional[int] = 256,
                scratch: bool = True, no_pool: bool = False, pretrained=False):
        super(CLAMP_resnet18, self).__init__()
        # backbone
        backbone = get_model(network_name, pretrain=pretrained)
        pool_layer = nn.Identity() if no_pool else None
        self.classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=bottleneck_dim,
                                        pool_layer=pool_layer, finetune=not scratch)
        domain_discri = DomainDiscriminator(in_feature=self.classifier.features_dim, hidden_size=bottleneck_dim)
        self.domain_adv = DomainAdversarialLoss(domain_discri)
        
        if dataset_name == DatasetsType.mnist or dataset_name == DatasetsType.usps:
            self.assessorS = assessor(h_dim=self.classifier.features_dim)
            self.assessorT = assessor(h_dim=self.classifier.features_dim)
        else:
            # lstm_nodes = 256
            lstm_nodes = 128
            # lstm_nodes = 64
            backbone = get_model('resnet18', pretrain=True)
            pool_layer = nn.Identity() if no_pool else None
            backbone = FeatrueExtrator(backbone, bottleneck_dim, pool_layer)
            self.assessorS = assessor_cnn(backbone=backbone, embed_dim=bottleneck_dim,
                                    hidden_size=lstm_nodes, num_layers=2, num_classes=3)
            self.assessorT = assessor_cnn(backbone=backbone, embed_dim=bottleneck_dim,
                                    hidden_size=lstm_nodes, num_layers=2, num_classes=3)

    def get_base_parameters(self) -> List[Dict]:
        param = self.classifier.get_parameters() + self.domain_adv.domain_discriminator.get_parameters()
        return param

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return