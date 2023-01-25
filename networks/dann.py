import torch.nn as nn

import torch.nn.functional as F
from utils import ReverseLayerF
from copy import deepcopy

class G_net(nn.Module):
    def __init__(self, init_net, bottleneck_dim, no_pool):
        """
        Arguments:
            init_net: backbone
            no_pool: pool layer
            bottleneck_dim: hidden layer size
        """
        super().__init__()
        
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        self.bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(init_net.out_features, bottleneck_dim),
            # nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.backbone = init_net

    def forward(self, x):
        """"""
        try:
            f = self.pool_layer(self.backbone(x))
        except:
            f = self.backbone(x)
        f = self.bottleneck(f)
        return f
    
class Classifier(nn.Module):
    """
    A 3-layer MLP for classification.
    """

    def __init__(self, num_classes, in_size=2048, h=1024):
        """
        Arguments:
            num_classes: size of the output
            in_size: size of the input
            h: hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, num_classes),
        )

    def forward(self, x):
        """"""
        x = self.net(x)
        return F.softmax(x)


class Discriminator(nn.Module):
    """
    A 2-layer MLP for domain classification.
    """

    def __init__(self, in_size=2048, h=2048, batch_norm=True, sigmoid=False):
        super().__init__()
        """
        Arguments:
            in_size: size of the input
            h: hidden layer size
            out_size: size of the output
        """
        self.h = h
        # self.net = nn.Sequential(
        #     nn.Linear(in_size, h),
        #     nn.ReLU(),
        #     # nn.Linear(h, h),
        #     # nn.ReLU(),
        #     nn.Linear(h, 2),
        #     nn.LogSoftmax(dim=1)
        # )
        # self.out_size = out_size
        
        self.h = h
        if batch_norm:
            self.net = nn.Sequential(
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Linear(h, 2)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_size, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 2)
            )
        self.out_size = 2

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.net(reversed_input)
        return F.softmax(x)


class DANN(nn.Module):
    def __init__(self, encoder: nn.Module, domain_classifier: nn.Module, classifier: nn.Module):
        super(DANN, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = domain_classifier

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Dann(nn.Module):
    def __init__(self, backbone, f=2048, h=2048, n_outputs=10):
        super(Dann, self).__init__()
        self.f = backbone
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        self.bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, f),
            nn.BatchNorm1d(f),
            nn.ReLU()
        )
        self.lc = nn.Sequential(
            nn.Linear(f, h),
            # nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(h, h),
            # nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, n_outputs),
        )
        self.dc = nn.Sequential(
            nn.Linear(f, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )
        self.bottleneck.apply(init_weights)
        self.lc.apply(init_weights)
    def forward(self, x,alpha):
        try:
            f = self.pool_layer(self.f(x))
        except:
            f = self.f(x)
        x = self.bottleneck(f)
        y = ReverseLayerF.apply(x, alpha)
        x = self.lc(x)
        y = self.dc(y)
        x=x.view(x.shape[0],-1)
        y=y.view(y.shape[0],-1)
        return x, y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return