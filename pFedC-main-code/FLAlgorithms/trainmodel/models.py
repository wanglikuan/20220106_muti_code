import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_

import collections

#################################
##### Neural Network model #####
#################################

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, dataset='mnist', model='cnn_Lenet', feature_dim=50*4*4, bottleneck_dim=256, num_labels=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 6, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            # nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 16, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(p=0.5),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(1),
        )
        # self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        # self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(784, 32)
        # self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(32, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        # x = self.bn(x)
        # x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class Net(nn.Module):
    def __init__(self, dataset='mnist', model='cnn_net'):
        super(Net, self).__init__()
        # define network layers
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[dataset]
        print('Network configs:', configs)
        self.named_layers, self.layers, self.layer_names =self.build_network(
            configs, input_channel, self.output_dim)
        self.n_parameters = len(list(self.parameters()))
        self.n_share_parameters = len(self.get_encoder())

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def build_network(self, configs, input_channel, output_dim):
        layers = nn.ModuleList()
        named_layers = {}
        layer_names = []
        kernel_size, stride, padding = 3, 2, 1
        for i, x in enumerate(configs):
            if x == 'F':
                layer_name='flatten{}'.format(i)
                layer=nn.Flatten(1)
                layers+=[layer]
                layer_names+=[layer_name]
            elif x == 'M':
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                layer_name = 'pool{}'.format(i)
                layers += [pool_layer]
                layer_names += [layer_name]
            else:
                cnn_name = 'encode_cnn{}'.format(i)
                cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
                named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]

                bn_name = 'encode_batchnorm{}'.format(i)
                bn_layer = nn.BatchNorm2d(x)
                named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]

                relu_name = 'relu{}'.format(i)
                relu_layer = nn.ReLU(inplace=True)# no parameters to learn

                layers += [cnn_layer, bn_layer, relu_layer]
                layer_names += [cnn_name, bn_name, relu_name]
                input_channel = x

        # finally, classification layer
        fc_layer_name1 = 'encode_fc1'
        fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
        layers += [fc_layer1]
        layer_names += [fc_layer_name1]
        named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]

        fc_layer_name = 'decode_fc2'
        fc_layer = nn.Linear(self.latent_dim, self.output_dim)
        layers += [fc_layer]
        layer_names += [fc_layer_name]
        named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
        return named_layers, layers, layer_names


    def get_parameters_by_keyword(self, keyword='encode'):
        params=[]
        for name, layer in zip(self.layer_names, self.layers):
            if keyword in name:
                #layer = self.layers[name]
                params += [layer.weight, layer.bias]
        return params

    def get_encoder(self):
        return self.get_parameters_by_keyword("encode")

    def get_decoder(self):
        return self.get_parameters_by_keyword("decode")

    def get_shared_parameters(self, detach=False):
        return self.get_parameters_by_keyword("decode_fc2")

    def get_learnable_params(self):
        return self.get_encoder() + self.get_decoder()

    def forward(self, x, start_layer_idx = 0, logit=False):
        """
        :param x:
        :param logit: return logit vector before the last softmax layer
        :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
        :return:
        """
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        restults={}
        z = x
        for idx in range(start_layer_idx, len(self.layers)):
            layer_name = self.layer_names[idx]
            layer = self.layers[idx]
            z = layer(z)

        if self.output_dim > 1:
            # restults['output'] = F.log_softmax(z, dim=1)
            restults = F.log_softmax(z, dim=1)
        else:
            # restults['output'] = z
            restults = z
        # if logit:
        #     restults['logit']=z
        return restults

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        n_layers = len(self.layers)
        for layer_idx in range(n_layers + start_layer_idx, n_layers):
            layer = self.layers[layer_idx]
            z = layer(z)
        if self.output_dim > 1:
            out=F.log_softmax(z, dim=1)
        # result = {'output': out}
        result = out
        # if logit:
        #     result['logit'] = z
        return result
