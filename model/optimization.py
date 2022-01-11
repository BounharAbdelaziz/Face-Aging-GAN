from model.block import *
import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler

import torch.nn.utils as tutils
from torch.optim import SGD, Adam, AdamW
import pytorch_warmup


def define_network(net, data_device, gpu_ids):
    
    if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net)
    net.to(data_device)
   
    return net
    
def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if isinstance(net, nn.DataParallel):
        network_name = net.module.__class__.__name__
    else:
        network_name = net.__class__.__name__

    print('initialize network %s with %s' % (network_name, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>

    return net

def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net 

def get_optimizer(model, options):

    if options.optimizer.upper() == "ADAMW":
      optimizer = AdamW(model.parameters(), lr=options.lr, betas=(options.beta1, 0.999), weight_decay=options.weight_decay)

    elif options.optimizer.upper() == "ADAM":
      optimizer = Adam(model.parameters(), betas=(options.beta1, 0.999), lr=options.lr)

    else:
      optimizer = SGD(model.parameters(), lr=options.lr, weight_decay=options.weight_decay, momentum=0.9)

    return optimizer

    
def get_scheduler(optimizer, options):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if options.lr_policy.upper() == 'LINEAR':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + options.epoch_count - options.n_epochs) / float(options.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif options.lr_policy.upper() == 'STEP':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=options.lr_decay_iters, gamma=0.1)

    elif options.lr_policy.upper() == 'PLATEAU':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif options.lr_policy.upper() == 'COSINE':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options.n_epochs, eta_min=0)

    elif options.lr_policy.upper() == "EXP":
        scheduler = lr_scheduler.ExponentialLR(optimizer, options.gamma)
    
    elif options.lr_policy.upper() == "CYCLICLR":
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=options.lr, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=options.gamma)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', options.lr_policy)
    return scheduler

def get_lr_warmup(optimizer, options):

    return pytorch_warmup.ExponentialWarmup(optimizer, options.warmup_period)


