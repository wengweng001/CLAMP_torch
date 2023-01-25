import torch
import torch.nn as nn
import numpy as np

from utils import ReverseLayerF
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50

class MLPEncoder(nn.Module):
    def __init__(self, image_size=28, channels=1, h_dim=256):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size * channels, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.activation = nn.ReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, outputs):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, outputs)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return self.relu(feat)

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class Office31Encoder(nn.Module):
    def __init__(self, h_dim=1024):
        super(Office31Encoder, self).__init__()
        self.encoder = resnet34(pretrained=True)
        self.fc = nn.Linear(self.encoder.fc.out_features, h_dim)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1,3,224,224)
        x = self.encoder(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

from backbone import network_dict
class OfficeEncoder(nn.Module):
    def __init__(self, network='resnet34', h_dim=1024):
        super(OfficeEncoder, self).__init__()
        self.encoder = network_dict[network]()
        self.fc = nn.Linear(self.encoder.output_num(), h_dim)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1,3,224,224)
        x = self.encoder(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class OfficeHomeEncoder(nn.Module):
    def __init__(self, h_dim=1024):
        super(OfficeHomeEncoder, self).__init__()
        self.encoder = resnet50(pretrained=True)
        self.fc = nn.Linear(self.encoder.fc.out_features, h_dim)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class ConvEncoder(nn.Module):
    def __init__(self, network='resnet34', actication=None):
        super(ConvEncoder, self).__init__()
        self.encoder = network_dict[network]()

        # if base_net=='resnet18' or base_net=='resnet34' or base_net='resnet50':
        if 'resnet' in network:
            bottleneck=1000
        elif network=='vgg16' or network=='alexnet':
            bottleneck=2048
            
        self.bottleneck = nn.Linear(self.encoder.output_num(), bottleneck)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.0)
        
        self.activation = False
        if actication is not None:
            self.activation = True
            self.relu       = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        if self.activation:
            x = self.relu(x)
        return x

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class Classifier(nn.Module):
    """Classifier model for CLAMP."""

    def __init__(self, inputs, outputs):
        """Init classifier."""
        super(Classifier, self).__init__()
        self.fc = nn.Linear(inputs, outputs)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.fc(feat)
        # out = F.softmax(out)
        return out

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class Classifier_multi(nn.Module):
    """Classifier model for CLAMP."""

    def __init__(self, inputs, nhid, outputs):
        """Init classifier."""
        super(Classifier_multi, self).__init__()
        self.fc1 = nn.Linear(inputs, nhid)
        self.fc2 = nn.Linear(nhid, outputs)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.zero_()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.fc1(feat)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

class DomainClassifier(nn.Module):
    """Discriminator model for input domain."""

    def __init__(self, no_input, output_dims=2):
        """Init domain classifier."""
        super(DomainClassifier, self).__init__()
        self.linear = nn.Linear(no_input, round(no_input/2), bias=True)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        self.classifier = nn.Linear(round(no_input/2), output_dims)

    def forward(self, x, alpha=1.0):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        out = self.linear(reverse_feature)
        out = self.activation(out)
        out = self.classifier(out)
        out = F.softmax(out,1)
        return out

class DomainClassifier_3layer(nn.Module):
    def __init__(self,no_input, no_hidden):
        super(DomainClassifier_3layer, self).__init__()
        self.linear1 = nn.Linear(no_input, no_hidden, bias=True)
        self.linear2 = nn.Linear(no_hidden, no_hidden, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(no_hidden, 2)
        self.dropput = nn.Dropout()
        # self.logsoftmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.zero_()

    def forward(self, x, alpha=1.0):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        x = self.linear1(reverse_feature)
        x = self.activation(x)
        x = self.dropput(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropput(x)
        x = self.linear3(x)
        # x = self.logsoftmax(x)
        return x
        
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

        h0 = torch.zeros(2, x.shape[0], 64).to(self.device)
        c0 = torch.zeros(2, x.shape[0], 64).to(self.device)

        out ,_ = self.lstm(x ,(h0 ,c0))
        out = out[: ,-1 ,:]
        out = self.fc3(out)
        out = self.fc4(out)
        output = torch.sigmoid(out)
        output = np.mean(output.cpu().detach().numpy() ,axis=0)
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

        h0 = torch.zeros(2, x.size(0), 128).to(self.device)
        c0 = torch.zeros(2, x.size(0), 128).to(self.device)

        out ,_ = self.lstm(x ,(h0 ,c0))
        out = out[: ,-1 ,:]
        out = self.fc1(out)
        out = self.fc2(out)
        output = torch.sigmoid(out)
        output = np.mean(output.cpu().detach().numpy() ,axis=0)
        return output

class assessor_cnn(nn.Module):
    """Assessor module for CLAMP."""

    def __init__(self, h_dim=1024, resnet_34=False, resnet_50=False):
        """Init assessor."""
        super(assessor_cnn, self).__init__()

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
        output = np.mean(output.cpu().detach().numpy() ,axis=0)
        return output
