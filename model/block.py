import torch.nn as nn

from torch.nn import Softmax, ReLU, LeakyReLU, Sigmoid, Tanh


# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
     
class ActivationLayer(nn.Module):

    def __init__(self, activation='lk_relu', alpha_relu=0.15):
        super().__init__()

        if activation =='lk_relu':
            self.activation = nn.LeakyReLU(alpha_relu)

        elif activation =='relu':
            self.activation = nn.ReLU()

        elif activation =='softmax':
            self.activation = nn.Softmax()

        elif activation =='sigmoid':
            self.activation = nn.Sigmoid()

        elif activation =='tanh':
            self.activation = nn.Tanh()

        else :
            # Identity function
            self.activation = None

    def forward(self, x):

        if self.activation is None :
            # Identity function
            return x
        
        else :
            return self.activation(x)

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class NormalizationLayer(nn.Module):

    def __init__(self, in_features, norm_type='bn'):
        super().__init__()

        if norm_type == 'bn2d' :
            self.norm = nn.BatchNorm2d(in_features)

        elif norm_type == 'in2d' :
            self.norm = nn.InstanceNorm2d(in_features)

        elif norm_type == 'bn1d' :
            self.norm = nn.BatchNorm1d(in_features)

        elif norm_type == 'in1d' :
            self.norm = nn.InstanceNorm1d(in_features)

        elif norm_type == 'none' :
            self.norm = lambda x : x * 1.0

        else:
            raise NotImplementedError('[INFO] The Normalization layer %s is not implemented !' % norm_type)

    def forward(self, x):
        out = self.norm(x)
        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, norm_type='bn', activation='lk_relu', alpha_relu=0.15, norm_before=True, use_bias=True):
        super().__init__()
        
        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before

        # Fully connected layer
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        # Activation layer
        if activation == 'lk_relu':
            self.activation = ActivationLayer(activation=activation, alpha_relu=alpha_relu)
        else :
            self.activation = ActivationLayer(activation=activation)

        # Normalization layer
        self.norm = NormalizationLayer(in_features=out_features, norm_type=norm_type)

    def forward(self, x):
        
        out = self.linear(x)

        if self.norm_before :
            out = self.norm(out)
            out = self.activation(out)

        else :
            out = self.activation(out)
            out = self.norm(out)
        
        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class Conv2DLayer(nn.Module):
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

class ConvResidualBlock(nn.Module):
    """ Residual blocks idea is to feed the output of one layer to another layer after a number of hops (generaly 2 to 3). Here we are using a hops of 2.
        It can be expressed in the form : F(x) + x, where x is the input and F modeling one or many layers.
        The benefits of using Residual Blocks is to overcome the vanishing gradients problem, and thus training very deep networks.
    """
    def __init__(self, in_features, out_features, kernel_size=3, scale='none', use_pad=True, use_bias=True, norm_type='bn', norm_before=True, activation='lk_relu', alpha_relu=0.15, interpolation_mode='nearest'):
        super().__init__()

        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before

        # Computing the shortcut, we can use convolution or only identity
        if scale == 'none' and in_features == out_features :
            self.identity = lambda x : x
        else :
            self.identity = Conv2DLayer(in_features=in_features, out_features=out_features, kernel_size=kernel_size, scale=scale, use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, 
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
        self.conv1 = Conv2DLayer(in_features=in_features, out_features=out_features, kernel_size=kernel_size, scale=scales[0], use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, 
                                 norm_before=norm_before, activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)

        self.conv2 = Conv2DLayer(in_features=out_features, out_features=out_features, kernel_size=kernel_size, scale=scales[1], use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, 
                                 norm_before=norm_before, activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)

    

    def forward(self, x):

        identity = self.identity(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity

        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

class LinearResidualBlock(nn.Module):
    """ Residual blocks idea is to feed the output of one layer to another layer after a number of hops (generaly 2 to 3). Here we are using a hops of 2.
        It can be expressed in the form : F(x) + x, where x is the input and F modeling one or many layers.
        The benefits of using Residual Blocks is to overcome the vanishing gradients problem, and thus training very deep networks.
    """
    def __init__(self, in_features, out_features, use_bias=True, norm_type='bn', norm_before=True, activation='lk_relu', alpha_relu=0.15):
        super().__init__()

        # Sometimes, doing normalization before activation helps stabilizing the training
        self.norm_before = norm_before

        # Computing the shortcut, we can use convolution or only identity
        if in_features == out_features :
            self.identity = lambda x : x
        else :
            self.identity = LinearLayer(in_features=in_features, out_features=out_features, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

        # Linear layers that defines the Residual Block
        self.linear1 = LinearLayer(in_features=in_features, out_features=out_features, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)
        self.linear2 = LinearLayer(in_features=out_features, out_features=out_features, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)
    

    def forward(self, x):


        identity = self.identity(x)
        out = self.linear1(x)
        out = self.linear2(out)
        out = out + identity

        return out

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
