
import torch

from model.base_nets.base_model import BaseModel
from model.loss import GenLoss, DiscLoss

from model.optimization import get_optimizer, get_scheduler, get_lr_warmup

class BaseModelGAN(BaseModel):

    def __init__(self, generator, discriminator, options, hyperparams) -> None:
        super(BaseModelGAN, self).__init__(experiment=options.experiment_name)
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.is_train = options.is_train
        self.hyperparams = hyperparams
        self.options = options

        self.eps = 1e-8

        # Loss functions
        self.loss_names = []
        self.loss_G = GenLoss(options.gan_type).to(self.hyperparams.device)
        self.loss_D = DiscLoss(options.gan_type).to(self.hyperparams.device)
        self.loss_names.append("loss_G")
        self.loss_names.append("loss_D")

        # Hyper-parameters
        self.lambda_disc = hyperparams.lambda_disc
        self.lambda_gen = hyperparams.lambda_gen

        # Optimizers
        self.opt_gen = get_optimizer(self.generator, options)
        self.opt_disc = get_optimizer(self.discriminator, options)

        # Learning Rate Scheduler
        self.scheduler_gen = get_scheduler(self.opt_gen, options)
        self.scheduler_disc = get_scheduler(self.opt_disc, options)

        # Learning Rate Warmup stage
        if options.warmup_period:
            self.warmup_scheduler_gen = get_lr_warmup(self.opt_gen, options)
            self.warmup_scheduler_disc = get_lr_warmup(self.opt_disc, options)
        
        # Gradient scaler
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()