import torch.nn as nn
from model.block import Conv2DLayer, ConvResidualBlock


class IDFGANGenerator(nn.Module):

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def __init__( self, 
                norm_type='in2d', 
                norm_before=True, 
                activation='lk_relu', 
                alpha_relu=0.15, 
                use_bias=True,
                min_features = 32, 
                max_features=256,
                n_inputs=3, 
                n_output_first = 64,                
                n_ages_classes=5, 
                down_steps=2, 
                bottleneck_size=2, 
                up_steps=2,
                use_pad=True, 
                interpolation_mode='nearest', 
                kernel_size=3,
                use_UNet_archi=1,
                is_debug=False,
              ):

    """ 
    The Generator is in an encoder-decoder shape, we start by a small lattent vector for which we increase the dimensions to the desired image dimension.
    It can also follow a U-Net architecture.
    """
    super(IDFGANGenerator, self).__init__()

    self.is_debug = is_debug


    # conditional GAN - we input also n_ages_classes features maps. 
    # It's the same idea of One-Hot vectors when working with Linear layers, here with conv2dn we inject them as feature maps.
    n_inputs = n_inputs + n_ages_classes # defines the latent vector dimensions

    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))
    
    # Use a UNet like architecture
    self.use_UNet_archi = use_UNet_archi
    self.down_steps = down_steps
    self.up_steps = up_steps

    if is_debug:
      print("------------- Generator -------------")

    ##########################################
    #####             Encoder             ####
    ##########################################

    self.encoder = []

    if is_debug:
      print("------------ Encoder ---------------")
      
      print(f"input layer...")
      print(f"n_inputs : {n_inputs}")
      print(f"n_output_first : {n_output_first}")
      print("---------------------------")

    # input layer

    self.encoder.append(
      ConvResidualBlock(in_features=n_inputs, out_features=n_output_first, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
    )
    
    n_inputs = n_output_first
    n_output = features_cliping(n_output_first * 2)

    

    for i in range(down_steps-1):

      if is_debug:
        print(f"n_inputs : {n_inputs}")
        print(f"n_output : {n_output}")
        print("---------------------------")

      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

      if i != down_steps-2 :
        n_inputs = features_cliping(n_inputs * 2)
        n_output = features_cliping(n_output * 2)
      
    self.encoder = nn.Sequential(*self.encoder)

    ##########################################
    #####            Bottleneck           ####
    ##########################################
    if is_debug:
      print("------------ Bottleneck ---------------")


    self.bottleneck = []
    for i in range(bottleneck_size):

      if is_debug:
        print(f"n_inputs : {n_output}")
        print(f"n_output : {n_output}")
        print("---------------------------")

      self.bottleneck.append(
        ConvResidualBlock(in_features=n_output, out_features=n_output, kernel_size=kernel_size, scale='none', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

    self.bottleneck = nn.Sequential(*self.bottleneck)

    ##########################################
    #####             Decoder             ####
    ##########################################

    self.decoder = []

    if is_debug:
      print("------------ Decoder ---------------")

    for i in range(up_steps):
      if i == 0 :
        n_inputs = n_output
      else :
        n_inputs = features_cliping(n_inputs // 2)
      
      n_output = features_cliping(n_output // 2)
      
      if is_debug:
        print(f"n_inputs : {n_inputs}")
        print(f"n_output : {n_output}")
        print("---------------------------")

      self.decoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='up', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

    self.decoder = nn.Sequential(*self.decoder)

    # output layer, will put the image to the RGB channels
    if is_debug:
      print("------------ Output Layer ---------------")
      print(f"n_inputs : {n_output}")
      print(f"n_output : 3")
      print("---------------------------")

    self.out_layer = Conv2DLayer(in_features=n_output, out_features=3, kernel_size=3, scale='none', use_pad=use_pad, use_bias=use_bias, norm_type='none', norm_before=norm_before, activation='tanh', alpha_relu=alpha_relu)

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def forward(self, x) :
    
    # if we are using a U-Net like architecture
    if self.use_UNet_archi :
      residuals_enc = []

      if self.is_debug:
        print(f'[INFO] input shape : {x.shape}')

        # encoder
        for i in range(self.down_steps+1): # add +1 since the encoder encapsulate the input layer
          x = self.encoder[i](x)
          residuals_enc.append(x) # save them for later use in the decoder
          if self.is_debug:
            print(f'[INFO] i={i} - encoder : {x.shape}')

        # bottleneck
        if self.is_debug:
          print(f'[INFO] shape after encoder : {x.shape}')
          print(f'[INFO] len residuals_enc : {len(residuals_enc)}')
        x = self.bottleneck(x)
        if self.is_debug:
          print(f'[INFO] shape after bottleneck : {x.shape}')

        # decoder
        n = len(residuals_enc)
        for i in range(self.down_steps):
          idx_residual = n - i - 1
          if self.is_debug:
            print(f'[INFO] i={i} - decoder : {self.decoder[i](x).shape}')

          if idx_residual >= 0 :
            x = self.decoder[i](x) + residuals_enc[idx_residual]
          else :
            x = self.decoder[i](x)
          if self.is_debug:
              print(f'[INFO] i={i} - decoder : {x.shape}')

        # out_layer
        out = self.out_layer(x)

    else :    
      if self.is_debug:
        print(f'[INFO] input shape : {x.shape}')

      out = self.encoder(x)

      if self.is_debug:
        print(f'[INFO] shape after encoder : {out.shape}')

      out = self.bottleneck(out)
      if self.is_debug:
        print(f'[INFO] shape after bottleneck : {out.shape}')

      out = self.decoder(out)
      if self.is_debug:
        print(f'[INFO] shape after decoder : {out.shape}')
        
      out = self.out_layer(out)
      if self.is_debug: 
        print(f'[INFO] shape after generator : {out.shape}')

    return out

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#