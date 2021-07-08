from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam


class GenLoss(nn.Module):
  
    def __init__(self, gan_type='vanilla' ):
        
        super(GenLoss, self).__init__()
        self.gan_type = gan_type

        # vanilla GAN, min_D max_G log(D(x)) + log(1 - D(G(z)))
        if self.gan_type == 'vanilla' :
            self.criterion = nn.BCELoss()

        # least square error
        elif self.gan_type == 'lsgan' :
            self.criterion = nn.MSELoss()
        
        # Wasserstein GAN tackles the problem of Mode Collapse and Vanishing Gradient. 
        elif self.gan_type == 'wgan':
            self.criterion = None

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)

    def __call__(self, disc_pred):

        ##########################################
        #####         Generator Loss          ####
        ##########################################
        
        if self.gan_type == 'vanilla' :

            # we want to min log(1 - D(G(z))) which is equivalent to max log(D(G(z))) and it's better in the begining of the training (better gradients).
            loss_G = self.criterion(disc_pred, torch.ones_like(disc_pred))

        return loss_G


class DiscLoss(nn.Module):
  
    def __init__(self, gan_type='vanilla' ):
        
        super(DiscLoss, self).__init__()
        self.gan_type = gan_type

        # vanilla GAN, min_D max_G log(D(x)) + log(1 - D(G(z)))
        if self.gan_type == 'vanilla' :
            self.criterion = nn.BCELoss()

        # least square error
        elif self.gan_type == 'lsgan' :
            self.criterion = nn.MSELoss()
        
        # Wasserstein GAN tackles the problem of Mode Collapse and Vanishing Gradient. 
        elif self.gan_type == 'wgan':
            self.criterion = None

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)

    def __call__(self, disc_real, disc_fake):

        ##########################################
        #####       Discriminator Loss        ####
        ##########################################

        if self.gan_type == 'vanilla' :

            # the loss on the real image in the batch
            loss_D_real = self.criterion(disc_real, torch.ones_like(disc_real))
            # the loss on the fake image in the batch
            loss_D_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
            # total discriminator loss          
            loss_D = (loss_D_real + loss_D_fake) / 2

        return loss_D


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def __call__(self, x):

        return 0

class AgeLoss(nn.Module):
    def __init__(self):
        super(AgeLoss, self).__init__()

    def __call__(self, x):

        return 0


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()

    def __call__(self, x):

        return 0