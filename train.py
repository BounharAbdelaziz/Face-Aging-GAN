import torch
import numpy as np

from data_preprocessing.data_generator import CACD2000Data
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.id_face_gan import IDFaceGAN
from model.generator import Generator
from model.discriminator import Discriminator
from model.networks import AgeClassifier
from model.optimization import init_weights, define_network
from utils.helpers import *

from pathlib import Path

# fix seeds for reproducibility
__seed__ = 42
torch.manual_seed(__seed__)
np.random.seed(__seed__)



# for batch_idx, real_data in enumerate(dataloader):
#     print(f'batch_idx : {batch_idx}')
#     print(f'batch size : {real_data.shape[0]}')
#     print(f'real_data.shape : {real_data.shape}')
#     if batch_idx == 0 :
#         break

batch_size=8
experiment="ID_FACE_GAN_v2_Lage_fnAgeCLF_wd_lr_sch_lsgan"

path = Path.cwd() / "check_points" / experiment
try :
    path.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f'[INFO] Checkpoint directory already exists')
else:
    print(f'[INFO] Checkpoint directory has been created')

## Init Models
hyperparams = Hyperparameters(  show_advance=5, 
                                lr=0.00003, 
                                batch_size=batch_size, 
                                n_epochs=100,
                                n_ages_classes=5,
                                save_weights=5000,
                                lambda_disc=1,
                                lambda_gen=1,
                                lambda_pcp=0.2,
                                lambda_age=30,
                                lambda_id=10,
                                lambda_mse=450,
                                num_threads=4)

# dumping hyperparams to keep track of them for later comparison/use
path_dump_hyperparams = path / "train_options.txt"
hyperparams.dump(path_dump_hyperparams)

## Init Dataloader
dataset = CACD2000Data(n_classes=hyperparams.n_ages_classes, h=256, w=256, n_classes_max=5)

# init dataloarder
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#, num_workers=hyperparams.num_threads) 

if batch_size > 1 :
    norm_type='bn2d'
else :
    norm_type='none'

generator = Generator(  norm_type=norm_type,
                        norm_before=True, 
                        activation='lk_relu', 
                        alpha_relu=0.2, 
                        use_bias=True,
                        min_features = 64, 
                        max_features=512,
                        n_inputs=3, 
                        n_output_first = 64,                
                        n_ages_classes=hyperparams.n_ages_classes, 
                        down_steps=2, 
                        bottleneck_size=4, 
                        up_steps=2,
                        use_pad=True, 
                        interpolation_mode='nearest', 
                        kernel_size=3,
                        use_UNet_archi=0,
                    ) # need to fix shapes problems for UNet archi

discriminator = Discriminator(norm_type=norm_type, n_ages_classes=hyperparams.n_ages_classes)


print(f"[INFO] Number of trainable parameters for the Generator : {compute_nbr_parameters(generator)}")
print(f"[INFO] Number of trainable parameters for the Discriminator : {compute_nbr_parameters(discriminator)}")
print("-----------------------------------------------------")

age_clf = AgeClassifier() # no init - pretrained network.
print(f"[INFO] Initializing the networks...")
generator=init_weights(generator, init_type='kaiming')
discriminator=init_weights(discriminator, init_type='kaiming')
print("-----------------------------------------------------")

device_ids = [i for i in range(torch.cuda.device_count())]
print(f"[INFO] Setting up {len(device_ids)} GPU(s) for the networks...")

generator=define_network(generator, hyperparams.device, device_ids)
discriminator=define_network(discriminator, hyperparams.device, device_ids)
age_clf=define_network(age_clf, hyperparams.device, device_ids)
print("-----------------------------------------------------")

network = IDFaceGAN(generator, discriminator, age_clf, hyperparams,experiment, gan_type='lsgan')

## Start Training

network.train(  dataloader=dataloader, 
                h=256, 
                w=256, 
                ckpt="./check_points/",
                max_step_train_age_clf=100.000,
                tune_age_clf=True)

## Save trained models