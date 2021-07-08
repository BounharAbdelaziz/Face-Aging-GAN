import torch.nn as nn
from torch.nn import Module, BCELoss, Linear, Conv2d
from torch.nn import Softmax, ReLU, LeakyReLU, Sigmoid, Tanh
from models.block import LinearLayer, ConvResidualBlock, LinearResidualBlock

class Discriminator(nn.Module):
  # -----------------------------------------------------------------------------#

  def __init__(   self, 
                  norm_type='in2d', 
                  norm_before=True, 
                  activation='lk_relu', 
                  alpha_relu=0.15, 
                  use_bias=True,
                  min_features = 16, 
                  max_features=256,
                  n_inputs=3, 
                  n_output = 64, 
                  output_dim=1,               
                  n_ages_classes=5, 
                  cgan=True, 
                  down_steps=2, 
                  use_pad=True, 
                  interpolation_mode='nearest', 
                  kernel_size=3,
              ):
    """
    The discriminator is in an encoder shape, we encode the features to a smaller space of features and do the decisions.
    """
    super(Discriminator, self).__init__()
    
    # conditional GAN - we input also the 10-dim features maps. We use the same idea of One-Hot vectors but inject them as feature maps.
    n_inputs = n_inputs + n_ages_classes # defines the latent vector dimensions

    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))

    ##########################################
    #####             Encoder             ####
    ##########################################

    self.encoder = []

    # input layer    
    self.encoder.append(
      ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                        activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
    )
    
    for i in range(down_steps):
      
      if i == 0 :
        n_inputs = n_output
        n_output = features_cliping(n_output // 2)

      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs // 2)
        n_output = features_cliping(n_output // 2)

    self.encoder = nn.Sequential(*self.encoder)

    self.flatten = nn.Flatten()

    self.out_layer = LinearLayer(in_features=n_inputs*n_output, out_features=output_dim, norm_type='none', activation='sigmoid', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#

  def forward(self, x) :
    out = self.encoder(x)
    out = self.flatten(out)
    out = self.out_layer(out)
    return out
    
  # -----------------------------------------------------------------------------#
  