from torch.utils.tensorboard import SummaryWriter
import torch

import torchvision
from torch.optim import SGD, Adam, AdamW

import utils.helpers as helper
from model.loss import L2Loss, GenLoss, DiscLoss, PerceptualLoss, AgeLoss, IDLoss

import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR, CosineAnnealingLR
import pytorch_warmup as warmup

class IDFaceGAN():
  
  def __init__(self, generator, discriminator, options, hyperparams, device_ids, num_steps, experiment="ID_FACE_GAN_v2_Lage", gan_type='lsgan', optimizer="AdamW", lr_scheduler="CosineAnnealingLR"):

    self.generator = generator
    self.discriminator = discriminator
    self.options = options

    # Optimizer
    if optimizer.upper() == "ADAMW":
      self.opt_gen = AdamW(self.generator.parameters(), lr=hyperparams.lr, betas=(0.9, 0.999), weight_decay=0.01)
      self.opt_disc = AdamW(self.discriminator.parameters(), lr=hyperparams.lr, betas=(0.9, 0.999), weight_decay=0.01)

    elif optimizer.upper() == "ADAM":
      self.opt_gen = Adam(self.generator.parameters(), betas=(0.9, 0.999), lr=hyperparams.lr)
      self.opt_disc = Adam(self.discriminator.parameters(), betas=(0.9, 0.999), lr=hyperparams.lr)

    else:
      self.opt_gen = SGD(self.generator.parameters(), lr=hyperparams.lr, weight_decay=0.01, momentum=0.9)
      self.opt_disc = SGD(self.discriminator.parameters(), lr=hyperparams.lr, weight_decay=0.01, momentum=0.9)

    # Learning Rate Scheduler

    if lr_scheduler.upper() == "EXP":
      self.scheduler_gen = ExponentialLR(self.opt_gen, gamma=0.75)
      self.scheduler_disc = ExponentialLR(self.opt_disc,  gamma=0.75)

    elif lr_scheduler.upper() == "COSINEANNEALINGLR":
      self.scheduler_gen = CosineAnnealingLR(self.opt_gen, T_max=num_steps)
      self.scheduler_disc = CosineAnnealingLR(self.opt_disc, T_max=num_steps)
      
    else:
      self.scheduler_gen = CyclicLR(self.opt_gen, base_lr=hyperparams.lr, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=0.85)
      self.scheduler_disc = CyclicLR(self.opt_disc, base_lr=hyperparams.lr, max_lr=0.1, step_size_up=5, mode="exp_range", gamma=0.85)

    # Learning Rate Warmup stage

    self.warmup_scheduler_gen = warmup.ExponentialWarmup(self.opt_gen, warmup_period=3000)
    self.warmup_scheduler_disc = warmup.ExponentialWarmup(self.opt_disc, warmup_period=3000)

    self.hyperparams = hyperparams

    # Loss functions
    self.loss_MSE = L2Loss(device=self.hyperparams.device)
    self.loss_GEN = GenLoss(gan_type).to(self.hyperparams.device)
    self.loss_DISC = DiscLoss(gan_type).to(self.hyperparams.device)
    self.loss_PCP = PerceptualLoss(device=self.hyperparams.device)
    self.loss_AGE = AgeLoss(self.hyperparams, device_ids).to(self.hyperparams.device)
    self.loss_ID = 0#IDLoss(device=self.hyperparams.device).to(self.hyperparams.device)

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
    loss_ID = 0#self.lambda_id*self.loss_ID(fake, real, y_val=1)
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


  def define_network(net, data_device, device_ids):
    if len(device_ids) > 0:
      assert(torch.cuda.is_available())
      net.to(data_device)
      net = torch.nn.DataParallel(net, device_ids, output_device=data_device)
    return net

  def train(self, dataloader):

    h = self.options.img_size
    w = self.options.img_size

    self.PATH_CKPT = self.options.checkpoints_dir + self.options.experiment_name+"/"
    
    print("[INFO] Started training using device ",self.hyperparams.device,"...")      
    
    step = 0
    for epoch in tqdm(range(self.hyperparams.n_epochs)):

      print("epoch = ",epoch," --------------------------------------------------------\n")
      
      for batch_idx, data in enumerate(dataloader) : 

        real_data, injected_age_class, _ = data

        with torch.autograd.set_detect_anomaly(True) :

          # real_data = real_data.to(self.hyperparams.device)
          real_data = real_data.float()
          real_data = real_data.view(self.hyperparams.batch_size, self.hyperparams.input_channels_gen, h, w).to(self.hyperparams.device)


          # we generate an image according to the age
          fake_data = self.generator(real_data)

          # The 5 feature maps that constitute the label
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
              
              # lr schedulers
              losses["lr_gen"] = self.scheduler_gen.get_last_lr()[0]
              losses["lr_disc"] = self.scheduler_disc.get_last_lr()[0]
              
              helper.write_logs_tb(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, img_fake, img_real, age_fake, losses, step, epoch, self.hyperparams, with_print_logs=True)

              step = step + real_data[0] # add batch size


          # Learning rate scheduler
          self.scheduler_gen.step(self.scheduler_gen.last_epoch+1)
          self.scheduler_disc.step(self.scheduler_disc.last_epoch+1)
          self.warmup_scheduler_gen.dampen()
          self.warmup_scheduler_disc.dampen()

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
