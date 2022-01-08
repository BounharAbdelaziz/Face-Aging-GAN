import torch

from datasets.UTKFace import UTKFaceData
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.cycle_gan.cycle_gan import CycleGAN
from model.cycle_gan.generator import CycleGANGenerator
from model.cycle_gan.discriminator import CycleGANDiscriminator


from utils.helpers import *

from pathlib import Path

from utils.helpers import fix_seed, create_checkpoints_dir
from options.training_opt import TrainOptions


# fix seeds for reproducibility
__seed__ = 42
fix_seed(__seed__)


experiment="FACE_GAN_cycleGAN"
PATH_AGE_CLF="./pretrained_models/Age_clf_Net_last_it_255400.pth"

path = Path.cwd() / "check_points" / experiment
create_checkpoints_dir(path)

## Init Models
options = TrainOptions().parse()
# options.print_options(options)

hyperparams = Hyperparameters(  show_advance=options.print_freq, 
                                lr=options.lr, 
                                batch_size=options.batch_size, 
                                n_epochs=options.n_epochs,
                                save_weights=options.save_weights_freq,
                                lambda_disc=options.lambda_D,
                                lambda_gen=options.lambda_G,
                                lambda_pcp=options.lambda_PCP,
                                lambda_id=options.lambda_ID,
                                lambda_mse=options.lambda_MSE,
                                num_threads=options.num_threads)

# dumping hyperparams to keep track of them for later comparison/use
path_dump_hyperparams = path / "train_options.txt"
hyperparams.dump(path_dump_hyperparams)

## Init Dataloader
dataset = UTKFaceData(options, domain_1_name="domain_1", domain_2_name="domain_2", do_transform=True)

# init dataloarder
dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

if options.batch_size > 1 :
    norm_type='bn2d'
else :
    norm_type='bn2d'


network = CycleGAN(hyperparams, options, experiment=options.experiment_name)

## Start Training
network.train(  dataloader=dataloader)