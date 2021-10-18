from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import SGD
import matplotlib.pyplot as plt

import utils.helpers as helper
from model.loss import GenLoss, DiscLoss, PerceptualLoss, AgeLoss, IDLoss

import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR

# os.environ['CUDA_VISIBLE_DEVICES']=2,3

class IDFaceGAN():
  
  def __init__(self, generator, discriminator, age_clf, hyperparams, experiment="ID_FACE_GAN_v2_Lage", gan_type='lsgan'):

    self.generator = generator
    self.discriminator = discriminator
    self.age_clf = age_clf

    # we use Adam optimizer for both Generator and Discriminator
    self.opt_gen = SGD(self.generator.parameters(), lr=hyperparams.lr, weight_decay=0.001, momentum=0.9)
    self.opt_disc = SGD(self.discriminator.parameters(), lr=hyperparams.lr, weight_decay=0.001, momentum=0.9)
    self.opt_age_clf = SGD(self.age_clf.parameters(), lr=hyperparams.lr_age_clf, weight_decay=0.0005, momentum=0.9)

    self.scheduler_gen = CyclicLR(self.opt_gen, base_lr=hyperparams.lr, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=0.85)
    self.scheduler_disc = CyclicLR(self.opt_disc, base_lr=hyperparams.lr, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=0.85)
    self.scheduler_age_clf = CyclicLR(self.opt_age_clf, base_lr=hyperparams.lr, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=0.85)

    self.hyperparams = hyperparams

    # Loss functions
    self.loss_MSE = nn.MSELoss().to(self.hyperparams.device)
    self.loss_GEN = GenLoss(gan_type).to(self.hyperparams.device)
    self.loss_DISC = DiscLoss(gan_type).to(self.hyperparams.device)
    self.loss_PCP = PerceptualLoss(device=self.hyperparams.device)
    self.loss_AGE = AgeLoss(device=self.hyperparams.device).to(self.hyperparams.device)
    self.loss_ID = IDLoss(device=self.hyperparams.device).to(self.hyperparams.device)

    self.eps = 1e-8

    # Hyper-parameters
    self.lambda_disc = hyperparams.lambda_disc
    self.lambda_gen = hyperparams.lambda_gen
    self.lambda_pcp = hyperparams.lambda_pcp
    self.lambda_age = hyperparams.lambda_age
    self.lambda_id = hyperparams.lambda_id
    self.lambda_mse = hyperparams.lambda_mse

    # Tensorboard logs
    self.experiment = experiment
    self.tb_writer_fake = SummaryWriter(f"logs/{self.experiment}_GAN/fake_{self.experiment}")
    self.tb_writer_real = SummaryWriter(f"logs/{self.experiment}_GAN/real_{self.experiment}")
    self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}_GAN/loss_train_{self.experiment}")

    
  def backward_G(self, disc_fake, fake, real,injected_age_class):

    loss_GEN = self.lambda_gen*self.loss_GEN(disc_fake)
    loss_PCP = self.lambda_pcp*self.loss_PCP(fake, real)
    loss_AGE = self.lambda_age*self.loss_AGE(fake, injected_age_class)
    loss_ID = self.lambda_id*self.loss_ID(fake, real, y_val=1)
    loss_MSE =  self.lambda_mse*self.loss_MSE(fake, real) + self.eps
   
    loss_G_total = loss_GEN + loss_PCP + loss_ID + loss_MSE + loss_AGE

    with torch.autograd.set_detect_anomaly(True) : # set to True only during debug
      loss_G_total.backward()

    return loss_G_total, loss_GEN, loss_PCP, loss_ID, loss_MSE, loss_AGE

  def backward_D(self, disc_real, disc_fake):

    loss_D = self.loss_DISC(disc_real, disc_fake)
    loss_D = self.lambda_disc*loss_D

    with torch.autograd.set_detect_anomaly(False) : # set to True only during debug
      loss_D.backward()

    return loss_D

  def backward_age_clf(self, logits, label):
    loss_age_clf = self.lambda_age*self.loss_AGE(logits, label)

    with torch.autograd.set_detect_anomaly(True) : # set to True only during debug
      loss_age_clf.backward()
    return loss_age_clf

  def optimize_network(self, disc_real, disc_fake, fake, real, real_age, pred_age, injected_age_class, tune_age_clf=False):
    
    # run backprop on the Discriminator
    self.opt_disc.zero_grad()
    loss_D = self.backward_D(disc_real, disc_fake)
    # Gradient steps for the Discriminator w.r.t its respective loss.
    self.opt_disc.step()

    # run backprop on the Generator
    self.opt_gen.zero_grad()
    loss_G_total, loss_GEN, loss_PCP, loss_ID, loss_MSE = self.backward_G(disc_fake, fake, real, injected_age_class)
    # Gradient steps for the Generator w.r.t its respective loss.
    self.opt_gen.step()

    # run backprop on the Age classifier
    if tune_age_clf :

      self.opt_age_clf.zero_grad()
      loss_age_clf = self.backward_age_clf(real_age, pred_age)
      # Gradient steps for the Age classifier w.r.t its respective loss.
      self.opt_age_clf.step()

      return loss_D, loss_G, loss_age_clf
    
    return loss_D, loss_G, loss_PCP, loss_ID

  def define_network(net, data_device, device_ids):
    if len(device_ids) > 0:
      assert(torch.cuda.is_available())
      net.to(data_device)
      net = torch.nn.DataParallel(net, device_ids, output_device=data_device)
    return net

  def train(self, dataloader, h=256, w=256, ckpt="./check_points/", max_step_train_age_clf=100.000, tune_age_clf=False):

    step = 0
    self.PATH_CKPT = ckpt+self.experiment+"/"
    
    print("[INFO] Started training using device ",self.hyperparams.device,"...")

    if self.hyperparams.device != 'cpu':
      # using DataParallel tu copy the Tensors on all available GPUs
      device_ids = [i for i in range(torch.cuda.device_count())]

      print(f'[INFO] Copying tensors to all available GPUs : {device_ids}')
      # should be reversed after the fix of dp

      
      # print(self.generator)
      # self.generator = self.define_network(net=self.generator, data_device=self.hyperparams.device, device_ids=device_ids)
      # self.discriminator = self.define_network(net=self.discriminator, data_device=self.hyperparams.device, device_ids=device_ids)
      # self.age_clf = self.define_network(net=self.age_clf, data_device=self.hyperparams.device, device_ids=device_ids)

      # if len(device_ids) > 1 :         
      #   self.generator = nn.DataParallel(self.generator)
      #   self.discriminator = nn.DataParallel(self.discriminator)
      #   self.age_clf = nn.DataParallel(self.age_clf)

      # self.generator.to(self.hyperparams.device)
      # self.discriminator.to(self.hyperparams.device)
      # self.age_clf.to(self.hyperparams.device)
      

    
    for epoch in tqdm(range(self.hyperparams.n_epochs)):

      print("epoch = ",epoch," --------------------------------------------------------\n")
      
      for batch_idx, data in enumerate(dataloader) : 

        real_data, injected_age_class, real_age_class = data
        # print(f'real_data device {real_data.get_device()}')
        # print(f'injected_age_class device {injected_age_class.get_device()}')
        # print(f'real_age_class device {real_age_class.get_device()}')

        if tune_age_clf:
          # print('optimizing age clf')
          # Optimizing the Age classifier
          # We train it first to have better predictions from the first iteration 
          self.opt_age_clf.zero_grad()
          loss_age_clf = self.backward_age_clf(real_data[:, :3, :, :], real_age_class)
          self.opt_age_clf.step()

          if batch_idx*epoch*len(dataloader.dataset) > max_step_train_age_clf:
            tune_age_clf = False

        # real_data = real_data.to(self.hyperparams.device)
        real_data = real_data.float()
        real_data = real_data.view(self.hyperparams.batch_size, self.hyperparams.input_channels_gen, h, w).to(self.hyperparams.device)


        # we generate an image according to the age
        # print("avant le gen")
        fake_data = self.generator(real_data)
        # print("après le gen")

        # The 5 feature maps that constitute the label
        fmap_age_lbl = real_data[:, 3:, :, :]
        # should do a column stack since it's the second dimensions now
        fake_data_clone = fake_data.clone()
        fake_data_disc = torch.column_stack((fake_data_clone, fmap_age_lbl))

        # prediction of the discriminator on real and fake images in the batch
        disc_real = self.discriminator(real_data)
        # detach from the computational graph to not re-use the output of the Generator
        disc_fake = self.discriminator(fake_data_disc.detach())

        real_data_opt = real_data[:, :3, :, :].clone()
        fake_data_opt = fake_data_disc[:, :3, :, :].clone()

        # Optimizing the Discriminator
        self.opt_disc.zero_grad()
        loss_D = self.backward_D(disc_real, disc_fake)
        self.opt_disc.step()

        # Optimizing the Generator
        disc_fake = self.discriminator(fake_data_disc)
        self.opt_gen.zero_grad()
        loss_G_total, loss_GEN, loss_PCP, loss_ID, loss_MSE, loss_AGE = self.backward_G(disc_fake, fake_data_opt, real_data_opt, injected_age_class)       
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

            losses = {}
            losses["loss_D"] = loss_D
            losses["loss_GEN"] = loss_GEN
            losses["loss_PCP"] = loss_PCP
            losses["loss_ID"] = loss_ID
            losses["loss_MSE"] = loss_MSE
            losses["loss_AGE"] = loss_AGE
            losses["loss_G_total"] = loss_G_total
            losses["loss_age_clf"] = loss_age_clf
            
            # lr schedulers
            losses["lr_gen"] = self.scheduler_gen.get_last_lr()[0]
            losses["lr_disc"] = self.scheduler_disc.get_last_lr()[0]
            losses["lr_age_clf"] = self.scheduler_age_clf.get_last_lr()[0]
            
            helper.write_logs_tb(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, img_fake, img_real, age_fake, losses, step, epoch, self.hyperparams, with_print_logs=True)

            step = step + 8

        if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

          # show advance
          print("[INFO] Saving weights...")
          # print(os.curdir)
          # print(os.path.abspath(self.PATH_CKPT))
          torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_it_"+str(step)+".pth"))
          torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_it_"+str(step)+".pth"))

      self.scheduler_gen.step()
      self.scheduler_disc.step()
      self.scheduler_age_clf.step()

    print("[INFO] Saving weights last step...")
    torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_last_it_"+str(step)+".pth"))
    torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_last_it_"+str(step)+".pth"))
    print(f'Latest networks saved in : {self.PATH_CKPT}')
