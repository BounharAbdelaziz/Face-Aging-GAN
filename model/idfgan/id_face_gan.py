from model.base_nets.base_model_gan import BaseModelGAN
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

import torchvision

import utils.helpers as helper
from model.loss import L2Loss, PerceptualLoss, AgeLoss, IDLoss

import os
from tqdm import tqdm

class IDFaceGAN(BaseModelGAN):

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#
  
  def __init__(self, generator, discriminator, options, hyperparams, device_ids):
    super(IDFaceGAN, self).__init__(generator, discriminator, options, hyperparams)

    # Loss functions
    if self.options.lambda_MSE:
        self.loss_MSE = L2Loss(device=self.hyperparams.device).to(self.hyperparams.device)
        self.loss_names.append("loss_MSE")

    if self.options.lambda_L1:
        self.loss_L1 = nn.L1Loss().to(self.hyperparams.device)    
        self.loss_names.append("loss_L1")

    if self.options.lambda_PCP:
        self.loss_PCP = PerceptualLoss(device=self.hyperparams.device).to(self.hyperparams.device)
        self.loss_names.append("loss_PCP")

    if self.options.lambda_ID:
        self.loss_ID = IDLoss(device=self.hyperparams.device).to(self.hyperparams.device)
        self.loss_names.append("loss_ID")

    if self.options.lambda_ID_cycle:
        self.loss_ID_cycle = nn.L1Loss().to(self.hyperparams.device)
        self.loss_names.append("loss_ID_cycle")

    if self.options.lambda_AGE:
      self.loss_AGE = AgeLoss(self.hyperparams, device_ids, is_debug=self.options.verbose).to(self.hyperparams.device)
      self.loss_names.append("loss_AGE")

      self.loss_cycle = nn.L1Loss().to(self.hyperparams.device)
      self.loss_l2_ = nn.MSELoss().to(self.hyperparams.device)
      print(f'[INFO] Using losses : {self.loss_names}')

    # Tensorboard logs
    self.experiment = options.experiment_name
    self.tb_writer_fake = SummaryWriter(f"logs/{self.experiment}_GAN/fake_{self.experiment}")
    self.tb_writer_real = SummaryWriter(f"logs/{self.experiment}_GAN/real_{self.experiment}")
    self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}_GAN/loss_train_{self.experiment}")

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#
    
  def backward_G(self, disc_fake, fake, real,injected_age_class):

    losses = {}

    loss_GEN = self.options.lambda_G * self.loss_GEN(disc_fake)
    losses["loss_GEN"] = loss_GEN

    loss_G_total = loss_GEN.item()
    
    if self.options.lambda_AGE:
      loss_AGE = self.options.lambda_AGE * self.loss_AGE(fake, injected_age_class) + self.eps
      loss_G_total = loss_G_total + loss_AGE
      losses["loss_AGE"] = loss_AGE


    if self.options.lambda_PCP:
      loss_PCP = self.lambda_PCP * self.loss_PCP(fake, real) + self.eps
      loss_G_total = loss_G_total + loss_PCP
      losses["loss_PCP"] = loss_PCP


    if self.options.lambda_ID:
      loss_ID = self.lambda_ID * self.loss_ID(fake, real, y_val=1) + self.eps
      loss_G_total = loss_G_total + loss_ID
      losses["loss_ID"] = loss_ID


    if self.options.lambda_MSE:
      loss_MSE =  self.options.lambda_MSE * self.loss_MSE(fake, real) + self.eps
      loss_G_total = loss_G_total + loss_MSE
      losses["loss_MSE"] = loss_MSE

    losses["loss_G_total"] = loss_G_total
   
    with torch.autograd.set_detect_anomaly(self.options.detect_anomaly) : # set to True only during debug
      loss_G_total.backward()

    return losses

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def backward_D(self, disc_real, disc_fake):

    loss_D = self.loss_DISC(disc_real, disc_fake)
    loss_D = self.options.lambda_D * loss_D + self.eps

    with torch.autograd.set_detect_anomaly(self.options.detect_anomaly) : # set to True only during debug
      loss_D.backward()

    return loss_D

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def get_last_lr(self, optimizer):
      return optimizer.param_groups[0]['lr']

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def train(self, dataloader):

    h = self.options.img_size
    w = self.options.img_size

    self.PATH_CKPT = self.options.checkpoints_dir + self.options.experiment_name+"/"
    
    print("[INFO] Started training using device ",self.hyperparams.device,"...")      
    
    step = 0
    for epoch in tqdm(range(self.hyperparams.n_epochs)):

      print("epoch = ",epoch," --------------------------------------------------------\n")
      
      for batch_idx, data in enumerate(dataloader) : 

        real_data, injected_age_class, real_age_class = data

        with torch.autograd.set_detect_anomaly(self.options.detect_anomaly) :

          real_data = real_data.float().to(self.hyperparams.device)
          # real_data = real_data.view(real_data.shape[0], self.hyperparams.input_channels_gen, h, w)


          # we generate an image according to the age
          fake_data = self.generator(real_data)

          # Extracting the feature maps that constitute the label
          fmap_age_lbl = real_data[:, 3:, :, :]
          
          # should do a column stack since it's the second dimensions now
          fake_data_clone = fake_data.clone()
          fake_data_disc = torch.column_stack((fake_data_clone, fmap_age_lbl))

          # prediction of the discriminator on real and fake images in the batch
          disc_real = self.discriminator( real_data[:, :3, :, :], fmap_age_lbl=fmap_age_lbl)
          # detach from the computational graph to not re-use the output of the Generator
          disc_fake = self.discriminator(fake_data_disc[:, :3, :, :].detach(), fmap_age_lbl=fmap_age_lbl)

          real_data_opt = real_data[:, :3, :, :].clone()
          fake_data_opt = fake_data_disc[:, :3, :, :].clone()

          # Optimizing the Discriminator
          self.opt_disc.zero_grad()
          loss_D = self.backward_D(disc_real, disc_fake)
          self.opt_disc.step()

          # Optimizing the Generator
          disc_fake = self.discriminator(fake_data_disc[:, :3, :, :], fmap_age_lbl=fmap_age_lbl)
          self.opt_gen.zero_grad()
          losses = self.backward_G(disc_fake, fake_data_opt, real_data_opt, injected_age_class)       
          self.opt_gen.step()

          # Logging advances

          if batch_idx % self.hyperparams.show_advance == 0 and batch_idx!=0:

            # show advance
            with torch.no_grad():
              
              age_fake = injected_age_class[0]
              fake_data_ = fake_data[0][:3, :, :].reshape(1, 3, h, w)
              real_data_ = real_data[0][:3, :, :].reshape(1, 3, h, w)

              img_fake = torchvision.utils.make_grid(fake_data_, normalize=True)
              img_real = torchvision.utils.make_grid(real_data_, normalize=True)

              losses["loss_D"] = loss_D
              
              # lr schedulers
              losses["lr_gen"] = self.get_last_lr(self.opt_gen)
              losses["lr_disc"] = self.get_last_lr(self.opt_disc)
              
              helper.write_logs_tb(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, img_fake, img_real, age_fake, losses, step, epoch, self.hyperparams, with_print_logs=True, experiment=self.options.experiment_name)

              step = step + batch_idx # add batch index


          # Learning rate scheduler
          self.scheduler_gen.step(self.scheduler_gen.last_epoch+1)
          self.scheduler_disc.step(self.scheduler_disc.last_epoch+1)
          if self.options.warmup_period:

            self.warmup_scheduler_gen.dampen()
            self.warmup_scheduler_disc.dampen()

        if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

          # Saving weights
          print("[INFO] Saving weights...")
          torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_it_"+str(step)+".pth"))
          torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_it_"+str(step)+".pth"))

          

    print("[INFO] Saving weights last step...")
    torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_last_it_"+str(step)+".pth"))
    torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_last_it_"+str(step)+".pth"))
    print(f'Latest networks saved in : {self.PATH_CKPT}')



    