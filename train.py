import torch
import numpy as np

from data_preprocessing.data_generator import CACD2000Data

from model.hyperparameters import Hyperparameters
from model.id_face_gan import IDFaceGAN
from model.generator import Generator
from model.discriminator import Discriminator
from model.networks import AgeClassifier

# fix seeds for reproducibility
__seed__ = 42
torch.manual_seed(__seed__)
np.random.seed(__seed__)

## Init Dataloader

dataloader = CACD2000Data()

for batch_idx, real_data in enumerate(dataloader):
    print(f'batch_idx : {batch_idx}')
    print(f'batch size : {real_data.shape[0]}')
    print(f'real_data.shape : {real_data.shape}')
    if batch_idx == 0 :
        break

## Init Models
hyperparams = Hyperparameters()
generator = Generator(  norm_type='none', 
                        down_steps=2, 
                        bottleneck_size=2, 
                        up_steps=3,
                        use_UNet_archi=0) # need to fix shapes problems for UNet archi
# print(generator)
discriminator = Discriminator(norm_type='none')
# print(discriminator)

age_clf = AgeClassifier()
network = IDFaceGAN(generator, discriminator, age_clf, hyperparams)

## Start Training

network.train(  dataloader=dataloader, 
                steps_train_disc=1, 
                h=256, 
                w=256, 
                print_loss_batch=True, 
                ckpt="./check_points/",
                max_step_train_age_clf=100)

## Save trained models