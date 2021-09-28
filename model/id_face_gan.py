from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt

import utils.helpers as helper
from model.loss import GenLoss, DiscLoss, PerceptualLoss, AgeLoss, IDLoss

import os
from tqdm import tqdm

class IDFaceGAN():
  
  def __init__(self, generator, discriminator, age_clf, hyperparams, experiment="ID_FACE_GAN_v1",):

    self.generator = generator
    self.discriminator = discriminator
    self.age_clf = age_clf

    # we use Adam optimizer for both Generator and Discriminator
    self.opt_gen = Adam(self.generator.parameters(), lr=hyperparams.lr)
    self.opt_disc = Adam(self.discriminator.parameters(), lr=hyperparams.lr)
    self.opt_age_clf = Adam(self.age_clf.parameters(), lr=hyperparams.lr)
    self.hyperparams = hyperparams

    # Loss functions
    self.loss_GEN = GenLoss()
    self.loss_DISC = DiscLoss()
    self.loss_PCP = PerceptualLoss()
    self.loss_AGE = AgeLoss()
    self.loss_ID = IDLoss()

    # Hyper-parameters
    self.lambda_disc = hyperparams.lambda_disc
    self.lambda_gen = hyperparams.lambda_gen
    self.lambda_pcp = hyperparams.lambda_pcp
    self.lambda_age = hyperparams.lambda_age
    self.lambda_id = hyperparams.lambda_id

    # Tensorboard logs
    self.experiment = experiment
    self.tb_writer_fake = SummaryWriter(f"logs/{self.experiment}_GAN/fake_{self.experiment}")
    self.tb_writer_real = SummaryWriter(f"logs/{self.experiment}_GAN/real_{self.experiment}")
    self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}_GAN/loss_train_{self.experiment}")

  
  def backward_G(self, disc_fake, fake, real):

    loss_GEN = self.loss_GEN(disc_fake)
    # loss_PCP = self.loss_PCP(fake, real)
    # loss_AGE = self.loss_AGE(fake, real)
    # loss_ID = self.loss_ID(fake, real, y_val=1)

    loss_G = self.lambda_gen*loss_GEN #+ self.lambda_pcp*loss_PCP  #+ self.lambda_id*loss_ID #+ self.lambda_age*loss_AGE

    with torch.autograd.set_detect_anomaly(True) : # set to True only during debug
      loss_G.backward()

    return loss_G

  def backward_D(self, disc_real, disc_fake):

    loss_D = self.loss_DISC(disc_real, disc_fake)
    loss_D = self.lambda_disc*loss_D

    with torch.autograd.set_detect_anomaly(True) : # set to True only during debug
      loss_D.backward(retain_graph=True)

    return loss_D

  def backward_age_clf(self, label, pred):
    loss_age_clf = 0
    with torch.autograd.set_detect_anomaly(True) : # set to True only during debug
      loss_age_clf.backward()
    return loss_age_clf

  def optimize_network(self, disc_real, disc_fake, fake, real, real_age, pred_age, tune_age_clf=False):
    
    # run backprop on the Discriminator
    self.opt_disc.zero_grad()
    loss_D = self.backward_D(disc_real, disc_fake)
    # Gradient steps for the Discriminator w.r.t its respective loss.
    self.opt_disc.step()

    # run backprop on the Generator
    self.opt_gen.zero_grad()
    loss_G = self.backward_G(disc_fake, fake, real)
    # Gradient steps for the Generator w.r.t its respective loss.
    self.opt_gen.step()

    # run backprop on the Age classifier
    if tune_age_clf :

      self.opt_age_clf.zero_grad()
      loss_age_clf = self.backward_age_clf(real_age, pred_age)
      # Gradient steps for the Age classifier w.r.t its respective loss.
      self.opt_age_clf.step()

      return loss_D, loss_G, loss_age_clf
    
    return loss_D, loss_G

  def train(self, dataloader, steps_train_disc=1, h=256, w=256, print_loss_batch=True, ckpt="./check_points/", max_step_train_age_clf=100):
    
    step = 0
    cpt = 0
    self.PATH_CKPT = ckpt+self.experiment+"/"
    
    print("[INFO] Started training using device ",self.hyperparams.device,"...")

    if self.hyperparams.device != 'cpu':
      # using DataParallel tu copy the Tensors on all available GPUs
      device_ids = [i for i in range(torch.cuda.device_count())]

      print(f'[INFO] Copying tensors to all available GPUs : {device_ids}')
      # if len(device_ids) > 1 :
      self.generator = nn.DataParallel(self.generator, device_ids)
      self.discriminator = nn.DataParallel(self.discriminator, device_ids)
      self.generator.to(self.hyperparams.device)
      self.discriminator.to(self.hyperparams.device)

    tune_age_clf = True
    for epoch in tqdm(range(self.hyperparams.n_epochs)):

      print("epoch = ",epoch," --------------------------------------------------------")

      for batch_idx, real_data in enumerate(dataloader) : 

        if batch_idx*epoch > max_step_train_age_clf:
          tune_age_clf = False

        real_data = real_data.view(1, self.hyperparams.input_channels_gen, h, w).to(self.hyperparams.device)
        batch_size = real_data.shape[0]

        print("[INFO] real_data.shape : ", real_data.shape)

        # we generate an image according to the age
        fake_data = self.generator(real_data)
        fmap_age_lbl = real_data[:, 3:, :, :]
        # should do a column stack since it's the second dimensions now
        fake_data = torch.column_stack((fake_data, fmap_age_lbl))

        # prediction of the discriminator on real an fake images in the batch
        disc_real = self.discriminator(real_data.float())
        # detach from the computational graph to not re-use the output of the Generator
        disc_fake = self.discriminator(fake_data.detach().float())#.detach()
        print("[INFO] disc_real : ", disc_real)
        print("[INFO] disc_fake : ", disc_fake)

        real_age = 10
        pred_age = 10
        
        real_data = real_data[:, :3, :, :]
        fake_data = fake_data[:, :3, :, :]
        tune_age_clf = 0
        if tune_age_clf :
          loss_D, loss_G, loss_age_clf = self.optimize_network(disc_real, disc_fake, fake_data, real_data, real_age, pred_age, tune_age_clf)
        else :
          loss_D, loss_G = self.optimize_network(disc_real, disc_fake, fake_data, real_data, real_age, pred_age, tune_age_clf)

        # Logging advances
        if print_loss_batch :
          print(f'[INFO] loss_D = {np.round(loss_D.item(), 4)}')
          print(f'[INFO] loss_G = {np.round(loss_G.item(), 4)}')

          if tune_age_clf :
            print(f'[INFO] loss_age_clf = {np.round(loss_age_clf.item(), 4)}')

        if batch_idx % self.hyperparams.show_advance == 0 and batch_idx!=0:

          # show advance
          print("[INFO] logging advance...")
          with torch.no_grad():

            # generate images
            fake_data_ = fake_data[0].reshape(-1, 3, h, w)
            real_data_ = real_data[0].reshape(-1, 3, h, w)

            img_fake = torchvision.utils.make_grid(fake_data_, normalize=True)
            img_real = torchvision.utils.make_grid(real_data_, normalize=True)

            helper.write_logs_tb(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, img_fake, img_real, loss_D, loss_G, step, epoch, self.hyperparams, with_print_logs=True)

            step = step + 1

        if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

          # show advance
          print("[INFO] Saving weights...")
          # print(os.curdir)
          # print(os.path.abspath(self.PATH_CKPT))
          torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_it_"+str(step)+".pth"))
          torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_it_"+str(step)+".pth"))

    print("[INFO] Saving weights last step...")
    torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_last_it_"+str(step)+".pth"))
    torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_last_it_"+str(step)+".pth"))
    print(f'Latest networks saved in : {self.PATH_CKPT}')
