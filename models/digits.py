"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn


class LeNet(nn.Sequential):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.num_classes = num_classes
        self.out_features = 500

    def copy_head(self):
        return nn.Linear(500, self.num_classes)


class DTN(nn.Sequential):
    def __init__(self, num_classes=10):
        super(DTN, self).__init__(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.num_classes = num_classes
        self.out_features = 512

    def copy_head(self):
        return nn.Linear(512, self.num_classes)



def lenet(pretrained=False, **kwargs):
    """LeNet model from
    `"Gradient-based learning applied to document recognition" <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 28 x 28.

    """
    return LeNet(**kwargs)


def dtn(pretrained=False, **kwargs):
    """ DTN model

    Args:
        num_classes (int): number of classes. Default: 10

    .. note::
        The input image size must be 32 x 32.

    """
    return DTN(**kwargs)

class mlp(nn.Module):
    def __init__(self, image_size=28, channels=1, h_dim=256, pretrained=False):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size * channels, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.activation = nn.ReLU()
        self.out_features = self.fc2.out_features

        nn.init.normal_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.normal_(self.fc2.weight)
        self.fc2.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc1.bias.data.zero_()
        # nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc2.bias.data.zero_()

    def forward(self, x):
        x = x.reshape(-1,784)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False