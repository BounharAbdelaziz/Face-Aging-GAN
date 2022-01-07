import torch 
import torch.nn as nn
from model.block import LinearLayer, ConvResidualBlock

class IDFGANDiscriminator(nn.Module):
  # -----------------------------------------------------------------------------#

  def __init__(   self, 
                  norm_type='in2d', 
                  norm_before=True, 
                  activation='lk_relu', 
                  alpha_relu=0.15, 
                  use_bias=True,
                  min_features = 32, 
                  max_features=512,
                  n_inputs=3, 
                  n_output = 64, 
                  age_fmap=None,
                  output_dim=1,               
                  n_ages_classes=5, 
                  down_steps=4, 
                  use_pad=True, 
                  interpolation_mode='nearest', 
                  kernel_size=3,
                  is_debug=False,
              ):
    """
      The discriminator is in an encoder shape, we encode the features to a smaller space of features and do the decisions.
        ## TO-DO : 
          * Add implementation of multi-scale discriminator.
    """
    super(IDFGANDiscriminator, self).__init__()
    
    
    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))

    self.is_debug = is_debug

    ##########################################
    #####             Encoder             ####
    ##########################################

    if is_debug:
      print("------------- Discriminator -------------")

    self.input_layer = []
    # input layer    
    self.input_layer.append(
      ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type='none', norm_before=norm_before, 
                        activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
    )
    if is_debug:
      print("------------- input layer -------------")

      print(f"n_inputs : {n_inputs}")
      print(f"n_output : {n_output}")

      print("------------- encoder -------------")
      
    # conditional GAN - we input also n_ages_classes features maps. 
    # It's the same idea of One-Hot vectors when working with Linear layers, here with conv2dn we inject them as feature maps.
    # to inject the information at the second layer of the discriminator
    n_output = n_output + n_ages_classes 
    self.encoder = []

    for i in range(down_steps-1):
      
      if i == 0 :
        n_inputs = n_output
        n_output = n_output - n_ages_classes
        n_output = features_cliping(n_output * 2)

      if is_debug:
        print(f"i : {i}")
        print(f"n_inputs : {n_inputs}")
        print(f"n_output : {n_output}")
        print("---------------------------")
      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

      if i != down_steps-1 :
        if i == 0 :
          n_inputs = features_cliping((n_inputs-n_ages_classes) * 2)
        else:
          n_inputs = features_cliping(n_inputs * 2)

        n_output = features_cliping(n_output * 2)

    self.input_layer = nn.Sequential(*self.input_layer)
    self.encoder = nn.Sequential(*self.encoder)

    self.flatten = nn.Flatten()

    self.out_layer = LinearLayer(in_features=n_output * 16 * 16, out_features=output_dim, norm_type='none', activation='lk_relu', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#

  def forward(self, x, fmap_age_lbl) :
    if self.is_debug:
      print(f'Discriminator input : {x.shape}')

    out = self.input_layer(x)
    if self.is_debug:
      print(f'input_layer output : {out.shape}')
      print(f'fmap_age_lbl[:,:, :128, :128] shape : {fmap_age_lbl[:,:, :128, :128].shape}')

    out = torch.column_stack((out, fmap_age_lbl[:,:, :128, :128]))

    if self.is_debug:
      print(f'encoder output : {out.shape}')

    out = self.encoder(out)
    if self.is_debug:
      print(f'encoder output : {out.shape}')

    out = self.flatten(out)    
    if self.is_debug:
      print(f'flatten output : {out.shape}')

    out = self.out_layer(out)
    if self.is_debug:
      print(f'out_layer output : {out.shape}')
    return out
    
  # -----------------------------------------------------------------------------#
  