
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class BaseModel(nn.Module):

    def __init__(self, hyperparams, experiment="train_dnn") -> None:
        super().__init__()
        # Tensorboard logs
        self.experiment = experiment
        self.hyperparams = hyperparams

        self.tb_writer_fake = SummaryWriter(f"logs/{self.experiment}_GAN/fake_{self.experiment}")
        self.tb_writer_real = SummaryWriter(f"logs/{self.experiment}_GAN/real_{self.experiment}")
        self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}_GAN/loss_train_{self.experiment}")

        self.eps = 1e-8
