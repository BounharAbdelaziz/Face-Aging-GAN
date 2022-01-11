import torch
import torch.nn as nn

from model.block import Conv2DLayer, ConvResidualBlock

class CycleGANGenerator(nn.Module):
    """ Generative network of CycleGAN"""
    
    def __init__(self, in_channels=3, out_channels=3, num_features=64, n_res_layers=9, norm_type="in2d", **kwargs) -> None:
        super().__init__()
        self.initial = Conv2DLayer( in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect", scale='none', use_pad=False, use_bias=True, norm_type='in2d', norm_before=True, 
                                    activation='relu', alpha_relu=0.15, inplace=True, **kwargs)
        
        self.down_blocks = nn.ModuleList(
            [
                Conv2DLayer( num_features, num_features*2, kernel_size=3, stride=2, padding=1, scale='down', use_pad=False, use_bias=True, norm_type=norm_type, norm_before=True, 
                                    activation='relu', inplace=True, **kwargs),

                Conv2DLayer( num_features*2, num_features*4, kernel_size=3, stride=2, padding=1, scale='down', use_pad=False, use_bias=True, norm_type=norm_type, norm_before=True, 
                                    activation='relu', inplace=True, **kwargs),
            ]
        )

        self.res_blocks = nn.Sequential(
            *[  ConvResidualBlock( num_features*4, num_features*4, kernel_size=3, scale='none', stride=1, use_pad=True, use_bias=True, norm_type=norm_type, norm_before=True, activation='relu', use_act_second=False,**kwargs)
                for _ in range(n_res_layers)
             ]
        )

        self.up_blocks = nn.ModuleList(
            [
                Conv2DLayer( num_features*4, num_features*2, kernel_size=3, stride=2, padding=0, scale='up', scale_factor=4, use_pad=True, use_bias=True, norm_type=norm_type, norm_before=True, 
                                    activation='relu', inplace=True, **kwargs),
                Conv2DLayer( num_features*2, num_features, kernel_size=3, stride=2, padding=0, scale='up', scale_factor=4, use_pad=True, use_bias=True, norm_type=norm_type, norm_before=True, 
                                    activation='relu', inplace=True, **kwargs),
            ]
        )

        self.last = nn.Conv2d(num_features*1, out_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):

        x = self.initial(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.res_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)
        
        return torch.tanh(self.last(x))