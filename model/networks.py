import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.functional as F

from model.block import LinearLayer, ConvResidualBlock, LinearResidualBlock, ActivationLayer,NormalizationLayer


# -----------------------------------------------------------------------------#
#                           Identity Preservation Module                       #
# -----------------------------------------------------------------------------#

"""                                                                                                     
    This implementation is based on this github repository : 
        https://github.com/AlfredXiangWu/LightCNN/tree/8b33107e836374a892efecd149d2016170167fdd 
"""

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, device='cuda', num_classes=80013):
        super(network_29layers_v2, self).__init__()

        self.conv1    = mfm(1, 48, 5, 1, 2).to(device)
        self.block1   = self._make_layer(block, layers[0], 48, 48).to(device)
        self.group1   = group(48, 96, 3, 1, 1).to(device)
        self.block2   = self._make_layer(block, layers[1], 96, 96).to(device)
        self.group2   = group(96, 192, 3, 1, 1).to(device)
        self.block3   = self._make_layer(block, layers[2], 192, 192).to(device)
        self.group3   = group(192, 128, 3, 1, 1).to(device)
        self.block4   = self._make_layer(block, layers[3], 128, 128).to(device)
        self.group4   = group(128, 128, 3, 1, 1).to(device)
        self.fc       = nn.Linear(8*8*128, 256).to(device)
        self.fc2 = nn.Linear(256, num_classes, bias=False).to(device)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        fc = self.fc(x)

        # we don't need to output the classification, only the embedding features are needed to compute the cosine similarity between two embedding vectors.

        return fc

def LightCNN_29Layers_v2(weights='./pretrained_models/LightCNN_29Layers_V2_checkpoint.pth', device='cuda', **kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], device, **kwargs)

    # load pretrained model weights
    ckpt = torch.load(weights, map_location=(lambda storage, loc : storage))

    # rename keys in state dictionnary since they are different. We remove the prefix "module."
    for key in list(ckpt['state_dict'].keys()):
        ckpt['state_dict'][key[7:]] = ckpt['state_dict'].pop(key)
    
    # load weights from dictionnary
    model.load_state_dict(ckpt['state_dict'])

    # let model on eval mode
    model.eval()
    model.to(device)

    return model

# -----------------------------------------------------------------------------#
#                     Perceptual Feature extraction Module                     #
# -----------------------------------------------------------------------------#

class VGG_19(nn.Module):

    def __init__(self, extraction_layer=''):
        super(VGG_19, self).__init__()
        self.vgg_net = models.vgg19(pretrained=True)
        self.extraction_layer = extraction_layer

    def forward(self, x):
        return x

# -----------------------------------------------------------------------------#
#                               Age Classifier                                 #
# -----------------------------------------------------------------------------#

class AgeClassifier(nn.Module):
  # -----------------------------------------------------------------------------#

  def __init__(   self, 
                  norm_type='bn2d', 
                  norm_before=True, 
                  activation='lk_relu', 
                  alpha_relu=0.2, 
                  use_bias=True,
                  min_features = 16, 
                  max_features=512,
                  n_inputs=3, 
                  n_output = 32, 
                  output_dim=5,               
                  down_steps=8, 
                  use_pad=True, 
                  kernel_size=3,
                  input_h=256,
                  input_w=256,
                  is_debug=False,
              ):
    """ The age classifier is in an encoder shape, we encode the features to a smaller space of features and do the decisions. """
    
    super(AgeClassifier, self).__init__()    
    self.is_debug = is_debug
    
    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))

    ##########################################
    #####             Encoder             ####
    ##########################################
    print(f"[INFO] ------------- Age classifier -------------")


    self.input_layer = []
    # input layer    
    self.input_layer.append(
      ConvResidualBlock2(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type='none', norm_before=norm_before, 
                        activation=activation, alpha_relu=alpha_relu)
    )
    if is_debug:
        print("------------- input layer -------------")

        print(f"n_inputs : {n_inputs}")
        print(f"n_output : {n_output}")

        print("------------- encoder -------------")
    self.encoder = []

    for i in range(down_steps-1):
      
      if i == 0 :
        n_inputs = n_output
        n_output = features_cliping(n_output * 2)

        if is_debug:
            print(f"i : {i}")
            print(f"n_inputs : {n_inputs}")
            print(f"n_output : {n_output}")
            print("---------------------------")

      self.encoder.append(
        ConvResidualBlock2(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, is_debug=is_debug)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs * 2)
        n_output = features_cliping(n_output * 2)

    self.input_layer = nn.Sequential(*self.input_layer)
    self.encoder = nn.Sequential(*self.encoder)

    self.flatten = nn.Flatten()
    
    new_h, new_w = input_h // (2**down_steps), input_w // (2**down_steps)
    flattened_dim = n_output*new_h*new_w
    
    self.residual_linears = []

    for i in range(4):
      self.residual_linears.append(
        LinearResidualBlock(  in_features=flattened_dim, out_features=flattened_dim, 
                              use_bias=use_bias, norm_type='bn1d', norm_before=norm_before, 
                              activation=activation, alpha_relu=alpha_relu))

    self.residual_linears = nn.Sequential(*self.residual_linears)
    # print("flattened_dim : ",flattened_dim)
    # print("n_output * 16 * 16 : ",n_output * 16 * 16)
    self.out_layer = LinearLayer(in_features=flattened_dim, out_features=output_dim, norm_type='none', 
                                activation='sigmoid', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#

  def forward(self, x) :
    if self.is_debug:
        print("---------------------------------------")
        print(f'AgeClassifier input : {x.shape}')

    out = self.input_layer(x)
    if self.is_debug:
        print(f'input_layer output : {out.shape}')

    out = self.encoder(out)
    if self.is_debug:
        print(f'encoder output : {out.shape}')

    out = self.flatten(out)   
    if self.is_debug:
        print(f'flatten output : {out.shape}')

    out = self.residual_linears(out)
    if self.is_debug:
        print(f'residual_linears output : {out.shape}')

    out = self.out_layer(out)
    if self.is_debug:
        print(f'out_layer output : {out.shape}')

    return out
    
  # -----------------------------------------------------------------------------#
  



class AgeClassifierAlexNet(nn.Module):
    """                                                                                                     
        Following the idea from https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf, 
        we finetune (change # of classes) AlexNet based on the CACD training set with 200.000 steps.
    """
    def __init__(self, n_ages_classes=5, backbone='alexnet', device='cuda'):
        super(AgeClassifier, self).__init__()
        
        # Parameters
        self.n_ages_classes = n_ages_classes
        self.backbone = backbone

        print(f"Using {backbone} as the Age classifier")

        # Network
        if backbone == 'alexnet':
            self.model = models.alexnet(pretrained=True).to(device)
            flattened_dim = 256 * 6 * 6

        elif backbone == 'resnet':
            self.model = models.resnet152(pretrained=True).to(device)
            flattened_dim = 1000

        self.flatten_layer = nn.Flatten().to(device)
        self.classifier = nn.Sequential(
                                    nn.Dropout(0.25),
                                    nn.Linear(flattened_dim, 4096),
                                    nn.ReLU(inplace=True),

                                    nn.Dropout(0.25),
                                    nn.Linear(4096, 4096),                
                                    nn.ReLU(inplace=True),

                                    nn.Dropout(0.25),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True),

                                    nn.Dropout(0.25),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True),

                                    nn.Linear(4096, self.n_ages_classes),
                                ).to(device)
        
    def forward(self, x):

        # Features extraction
        if self.backbone == 'alexnet':
            x = self.model.features(x)
            x = self.model.avgpool(x)

        elif self.backbone == 'resnet':
            x = self.model(x)

        x = self.flatten_layer(x)

        # logits
        x = self.classifier(x)

        return x


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
class Conv2DLayer2(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, scale='none', use_pad=True, use_bias=True, norm_type='bn', norm_before=True, activation='lk_relu', alpha_relu=0.15, interpolation_mode='nearest'):
        super().__init__()
        
        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before
        self.use_pad = use_pad

        # upsampling or downsampling
        stride = 2 if scale == 'down' else 1

        if scale == 'up':
            self.scale_layer = lambda x : nn.functional.interpolate(x, scale_factor=2, mode=interpolation_mode)
        else :
            self.scale_layer = lambda x : x

        # Padding layer
        self.padding = nn.ReflectionPad2d(kernel_size // 2) 

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride, bias=use_bias)

        # Activation layer
        if activation == 'lk_relu':
            self.activation = ActivationLayer(activation=activation, alpha_relu=alpha_relu)
        else :
            self.activation = ActivationLayer(activation=activation)

        # Normalization layer
        self.norm = NormalizationLayer(in_features=out_features, norm_type=norm_type)

    def forward(self, x):
        
        # upsampling or downsampling 
        out = self.scale_layer(x)

        if self.use_pad :
            out = self.padding(out)

        out = self.conv(out)

        if self.norm_before :
            out = self.norm(out)
            out = self.activation(out)

        else :
            out = self.activation(out)
            out = self.norm(out)

        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class ConvResidualBlock2(nn.Module):
    """ Residual blocks idea is to feed the output of one layer to another layer after a number of hops (generaly 2 to 3). Here we are using a hops of 2.
        It can be expressed in the form : F(x) + x, where x is the input and F modeling one or many layers.
        The benefits of using Residual Blocks is to overcome the vanishing gradients problem, and thus training very deep networks.
    """
    def __init__(self, in_features, out_features, kernel_size=3, scale='none', use_pad=True, use_bias=True, norm_type='bn', norm_before=True, activation='lk_relu', alpha_relu=0.15, interpolation_mode='nearest', is_debug=False):
        super().__init__()

        self.is_debug = is_debug
        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before

        # Computing the shortcut, we can use convolution or only identity
        if scale == 'none' and in_features == out_features :
            self.identity = lambda x : x
        else :
            self.identity = Conv2DLayer2(in_features=in_features, out_features=out_features, kernel_size=kernel_size, scale=scale, use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, 
                                 norm_before=norm_before, activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)

        # defining the scaling, we don't want to do 2 up sampling or 2 downsampling in one block, therefore :
        if scale == 'none' :
            scales = ['none', 'none']

        if scale == 'up' :
            # start by upsampling
            scales = ['up', 'none']

        if scale == 'down' :
            # downsampling in the end
            scales = ['none', 'down']

        # Convolutional layers that defines the Residual Block
        self.conv1 = Conv2DLayer2(in_features=in_features, out_features=out_features, kernel_size=kernel_size, scale=scales[0], use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, 
                                 norm_before=norm_before, activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)

        self.conv2 = Conv2DLayer2(in_features=out_features, out_features=out_features, kernel_size=kernel_size, scale=scales[1], use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, 
                                 norm_before=norm_before, activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)

    

    def forward(self, x):

        # identity = self.identity(x)
        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = out + identity

        if self.is_debug:
            print("---------------------------------------")

            print(f'ConvResidualBlock input : {x.shape}')

        identity = self.identity(x)
        if self.is_debug:
            print(f'identity shape : {identity.shape}')

        out = self.conv1(x)
        if self.is_debug:
            print(f'conv1 out : {out.shape}')


        out = self.conv2(out)
        if self.is_debug:
            print(f'conv2 out : {out.shape}')


        out = out + identity
        if self.is_debug:
            print(f'out : {out.shape}')

        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
