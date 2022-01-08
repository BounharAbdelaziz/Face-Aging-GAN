from model.base_nets.base_model import BaseModel
from model.base_nets.base_model_gan import BaseModelGAN
import torch
import torch.nn as nn
import torchvision

from model.loss import L2Loss, PerceptualLoss, IDLoss
from model.cycle_gan.generator import CycleGANGenerator
from model.cycle_gan.discriminator import CycleGANDiscriminator

import utils.helpers as helper
from tqdm import tqdm
import os
from model.optimization import get_optimizer, get_scheduler, get_lr_warmup

from torch.utils.tensorboard import SummaryWriter

class CycleGAN(BaseModel):

    """
        This class implements the CycleGAN model, for learning image-to-image translation without paired data.

        * CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def __init__(self, hyperparams, options, experiment="train_dnn") -> None:
        super(CycleGAN, self).__init__()
        self.hyperparams = hyperparams
        self.options = options
        self.loss_names = []
        self.loss_names.append("loss_G")
        self.loss_names.append("loss_D")

        self.experiment = experiment

        # Gradient scaler
        self.g1_scaler = torch.cuda.amp.GradScaler()
        self.g2_scaler = torch.cuda.amp.GradScaler()
        self.d1_scaler = torch.cuda.amp.GradScaler()
        self.d2_scaler = torch.cuda.amp.GradScaler()
        # self.hyperparams = hyperparams

        # Tensorboard
        self.tb_writer_fake = SummaryWriter(f"logs/{self.experiment}_GAN/fake_{self.experiment}")
        self.tb_writer_real = SummaryWriter(f"logs/{self.experiment}_GAN/real_{self.experiment}")
        self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}_GAN/loss_train_{self.experiment}")

        # Generators
        self.gen_1 = CycleGANGenerator(in_channels=3, n_res_layers=9).to(self.hyperparams.device)
        self.gen_2 = CycleGANGenerator(in_channels=3, n_res_layers=9).to(self.hyperparams.device)

        # Discriminators
        self.disc_1 = CycleGANDiscriminator(in_channels=3).to(self.hyperparams.device)
        self.disc_2 = CycleGANDiscriminator(in_channels=3).to(self.hyperparams.device)

        helper.setup_network(self.gen_1, self.hyperparams, type="generator")
        helper.setup_network(self.gen_2, self.hyperparams, type="generator")
        helper.setup_network(self.disc_1, self.hyperparams, type="discriminator")
        helper.setup_network(self.disc_2, self.hyperparams, type="discriminator")
        

        # Optimizers
        self.opt_disc_1 = get_optimizer(self.disc_1, self.options)
        self.opt_disc_2 = get_optimizer(self.disc_2, self.options)
        self.opt_gen_1 = get_optimizer(self.gen_1, self.options)
        self.opt_gen_2 = get_optimizer(self.gen_2, self.options)

        # Learning Rate Scheduler
        self.scheduler_g1 = get_scheduler(self.opt_gen_1, options)
        self.scheduler_g2 = get_scheduler(self.opt_gen_2, options)
        self.scheduler_d1 = get_scheduler(self.opt_disc_1, options)
        self.scheduler_d2 = get_scheduler(self.opt_disc_2, options)

        # Learning Rate Warmup stage
        if options.warmup_period:
            self.warmup_scheduler_g1 = get_lr_warmup(self.opt_gen_1, options)
            self.warmup_scheduler_g2 = get_lr_warmup(self.opt_gen_2, options)
            self.warmup_scheduler_d1 = get_lr_warmup(self.opt_disc_1, options)
            self.warmup_scheduler_d2 = get_lr_warmup(self.opt_disc_2, options)

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

        self.loss_cycle = nn.L1Loss().to(self.hyperparams.device)
        print(f'[INFO] Using losses : {self.loss_names}')

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, dataloader):
        loop = tqdm(dataloader, leave=True)

        h = self.options.img_size
        w = self.options.img_size

        self.PATH_CKPT = self.options.checkpoints_dir + self.options.experiment_name+"/"
        
        print("[INFO] Started training using device ",self.hyperparams.device,"...")      

        step = 0
        for epoch in tqdm(range(self.hyperparams.n_epochs)):
            print("epoch = ",epoch," --------------------------------------------------------\n")

            for batch_idx, data in enumerate(dataloader):
                
                img_1, img_2 = data
                img_1 = img_1.to(self.hyperparams.device)
                img_2 = img_2.to(self.hyperparams.device)

                # Train Discriminators 1 and 2
                with torch.cuda.amp.autocast():
                    # From domain 1 to domain 2
                    fake_img_1 = self.gen_1(img_2) # takes an image in domain 2 and generates one in domain 1
                    D_1_real = self.disc_1(img_1) # discriminates real/fake images in domain 2
                    D_1_fake = self.disc_1(fake_img_1.detach())
                    # Least-square GAN by default
                    D_1_real_loss = self.loss_MSE(D_1_real, torch.ones_like(D_1_real))
                    D_1_fake_loss = self.loss_MSE(D_1_fake, torch.zeros_like(D_1_fake))
                    D_1_loss = D_1_real_loss + D_1_fake_loss

                    # From domain 2 to domain 1
                    fake_img_2 = self.gen_2(img_1) # takes an image in domain 1 and generates one in domain 2
                    D_2_real = self.disc_2(img_2) # discriminates real/fake images in domain 1
                    D_2_fake = self.disc_2(fake_img_2.detach())
                    D_2_real_loss = self.loss_MSE(D_2_real, torch.ones_like(D_2_real))
                    D_2_fake_loss = self.loss_MSE(D_2_fake, torch.zeros_like(D_2_fake))
                    D_2_loss = D_2_real_loss + D_2_fake_loss


                self.opt_disc_1.zero_grad()
                self.d1_scaler.scale(D_1_loss).backward()
                self.d1_scaler.step(self.opt_disc_1)
                self.d1_scaler.update()

                self.opt_disc_2.zero_grad()
                self.d2_scaler.scale(D_2_loss).backward()
                self.d2_scaler.step(self.opt_disc_2)
                self.d2_scaler.update()


                # self.opt_disc_1.zero_grad()
                # D_1_loss.backward()
                # self.opt_disc_1.step()

                # self.opt_disc_2.zero_grad()
                # D_2_loss.backward()
                # self.opt_disc_2.step()


                # Train Generators 1 and 2
                with torch.cuda.amp.autocast():
                    # adversarial loss for both generators
                    D_1_fake = self.disc_1(fake_img_1) # discriminates real/fake images in domain 2
                    D_2_fake = self.disc_2(fake_img_2) # discriminates real/fake images in domain 1
                    # Least-square GAN by default
                    loss_G_1 = self.loss_MSE(D_1_fake, torch.ones_like(D_1_fake))
                    loss_G_2 = self.loss_MSE(D_2_fake, torch.ones_like(D_2_fake))

                    # cycle loss
                    cycle_1 = self.gen_1(fake_img_2) # G1 takes a fake image in domain 1 and generates back the original input image in domain 2
                    cycle_2 = self.gen_2(fake_img_1) # G2 takes a fake image in domain 1 and generates back the original input image in domain 2
                    cycle_1_loss = self.loss_cycle(img_2, cycle_1)
                    cycle_2_loss = self.loss_cycle(img_1, cycle_2)

                    loss_G1_total = loss_G_1 + cycle_1_loss * self.options.lambda_cycle
                    loss_G2_total = loss_G_2 + cycle_2_loss * self.options.lambda_cycle

                    # identity loss
                    if self.options.lambda_ID_cycle:
                        identity_1 = self.gen_1(img_2)
                        identity_2 = self.gen_2(img_1)
                        identity_1_loss = self.loss_ID_cycle(img_2, identity_1)
                        identity_2_loss = self.loss_ID_cycle(img_1, identity_2)
                        loss_G1_total = loss_G1_total + identity_1_loss * self.options.lambda_ID_cycle
                        loss_G2_total = loss_G2_total + identity_2_loss * self.options.lambda_ID_cycle
                    
                    # Loss functions
                    if self.options.lambda_MSE:
                        loss_MSE_1 =  self.options.lambda_MSE*self.loss_MSE(fake_img_1, img_1) + self.eps
                        loss_MSE_2 =  self.options.lambda_MSE*self.loss_MSE(fake_img_2, img_2) + self.eps
                        loss_G1_total = loss_G1_total + loss_MSE_1
                        loss_G2_total = loss_G2_total + loss_MSE_2

                    if self.options.lambda_L1:
                        loss_L1_1 =  self.options.lambda_L1*self.loss_L1(fake_img_1, img_1) + self.eps
                        loss_L1_2 =  self.options.lambda_L1*self.loss_L1(fake_img_2, img_2) + self.eps
                        loss_G1_total = loss_G1_total + loss_L1_1
                        loss_G2_total = loss_G2_total + loss_L1_2

                    if self.options.lambda_PCP:
                        loss_PCP_1 = self.options.lambda_PCP*self.loss_PCP(fake_img_1, img_1)
                        loss_PCP_2 = self.options.lambda_PCP*self.loss_PCP(fake_img_2, img_2)
                        loss_G1_total = loss_G1_total + loss_PCP_1
                        loss_G2_total = loss_G2_total + loss_PCP_2


                    if self.options.lambda_ID:
                        loss_ID_1 = self.options.lambda_ID*self.loss_ID(fake_img_1, img_1, y_val=1)
                        loss_ID_2 = self.options.lambda_ID*self.loss_ID(fake_img_2, img_2, y_val=1)
                        loss_G1_total = loss_G1_total + loss_ID_1
                        loss_G2_total = loss_G2_total + loss_ID_2

                # print("loss_G1_total :",loss_G1_total)
                # print("D_1_loss :",loss_G2_total)
                # print("loss_G2_total :",loss_G2_total)
                # print("D_2_loss :",loss_G2_total)

                self.opt_gen_1.zero_grad()
                self.g1_scaler.scale(loss_G1_total).backward(retain_graph=True)
                self.g1_scaler.step(self.opt_gen_1)
                self.g1_scaler.update()

                self.opt_gen_2.zero_grad()
                self.g2_scaler.scale(loss_G2_total).backward()
                self.g2_scaler.step(self.opt_gen_2)
                self.g2_scaler.update()

                # self.opt_gen_1.zero_grad()
                # loss_G1_total.backward()
                # self.opt_gen_1.step()

                # self.opt_gen_2.zero_grad()
                # loss_G2_total.backward()
                # self.opt_gen_2.step()

                
                # Learning rate scheduler
                self.scheduler_g1.step(self.scheduler_g1.last_epoch+1)
                self.scheduler_g2.step(self.scheduler_g2.last_epoch+1)
                self.scheduler_d1.step(self.scheduler_d1.last_epoch+1)
                self.scheduler_d2.step(self.scheduler_d2.last_epoch+1)

                if self.options.warmup_period:
                    self.warmup_scheduler_g1.dampen()
                    self.warmup_scheduler_g2.dampen()
                    self.warmup_scheduler_d1.dampen()
                    self.warmup_scheduler_d2.dampen()

                # Logging advances
                if batch_idx % self.hyperparams.show_advance == 0 and batch_idx!=0:

                    # show advance in tensorboard
                    with torch.no_grad():
                    
                        fake_img_1_ = fake_img_1[0][:3, :, :].reshape(1, 3, h, w)
                        fake_img_2_ = fake_img_2[0][:3, :, :].reshape(1, 3, h, w)

                        real_img_1_ = img_1[0][:3, :, :].reshape(1, 3, h, w)
                        real_img_2_ = img_2[0][:3, :, :].reshape(1, 3, h, w)

                        cycle_1_ = cycle_1[0][:3, :, :].reshape(1, 3, h, w)
                        cycle_2_ = cycle_2[0][:3, :, :].reshape(1, 3, h, w)

                        img_fake_1 = torchvision.utils.make_grid(fake_img_1_, normalize=True)
                        img_fake_2 = torchvision.utils.make_grid(fake_img_2_, normalize=True)

                        img_cycle_1 = torchvision.utils.make_grid(cycle_1_, normalize=True)
                        img_cycle_2 = torchvision.utils.make_grid(cycle_2_, normalize=True)

                        img_real_1 = torchvision.utils.make_grid(real_img_1_, normalize=True)
                        img_real_2 = torchvision.utils.make_grid(real_img_2_, normalize=True)
                        
                        images_fakes = [img_fake_1, img_fake_2, img_cycle_1, img_cycle_2]
                        images_real = [img_real_1, img_real_2]

                        losses = {}
                        # Loss functions
                        if self.options.lambda_MSE:
                            losses["loss_MSE_1"] = loss_MSE_1
                            losses["loss_MSE_2"] = loss_MSE_2


                        if self.options.lambda_L1:
                            losses["loss_L1_1"] = loss_L1_1
                            losses["loss_L1_2"] = loss_L1_2

                        if self.options.lambda_PCP:
                            losses["loss_PCP_1"] = loss_PCP_1
                            losses["loss_PCP_2"] = loss_PCP_2

                        if self.options.lambda_ID:
                            losses["loss_ID_1"] = loss_ID_1
                            losses["loss_ID_2"] = loss_ID_2

                        if self.options.lambda_ID_cycle:
                            losses["loss_ID_cycle_1"] = identity_1_loss
                            losses["loss_ID_cycle_2"] = identity_2_loss


                        losses["loss_D1"] = D_1_loss
                        losses["loss_D2"] = D_2_loss
                        losses["loss_G1"] = loss_G_1
                        losses["loss_G2"] = loss_G_2
                        losses["loss_G1_total"] = loss_G1_total
                        losses["loss_G2_total"] = loss_G2_total
                        
                        # lr schedulers
                        def get_last_lr_bis(optimizer):
                            return optimizer.param_groups[0]['lr']
                        losses["lr_g1"] = get_last_lr_bis(self.opt_gen_1) #self.scheduler_g1.get_last_lr()[0]
                        losses["lr_d1"] = get_last_lr_bis(self.opt_disc_1) #self.scheduler_d1.get_last_lr()[0]
                        losses["lr_g2"] = get_last_lr_bis(self.opt_gen_2) #self.scheduler_g2.get_last_lr()[0]
                        losses["lr_d2"] = get_last_lr_bis(self.opt_disc_2) #self.scheduler_d2.get_last_lr()[0]
                        
                        helper.write_logs_tb_cyclegan(self.tb_writer_loss, self.tb_writer_fake, self.tb_writer_real, images_fakes, images_real, losses, step, epoch, self.hyperparams, with_print_logs=True)

                        step = step + batch_idx


                if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

                    # Saving weights
                    print("[INFO] Saving weights...")
                    torch.save(self.disc_1.state_dict(), os.path.join(self.PATH_CKPT,"D1_it_"+str(step)+".pth"))
                    torch.save(self.disc_2.state_dict(), os.path.join(self.PATH_CKPT,"D2_it_"+str(step)+".pth"))
                    torch.save(self.gen_1.state_dict(), os.path.join(self.PATH_CKPT,"G1_it_"+str(step)+".pth"))
                    torch.save(self.gen_2.state_dict(), os.path.join(self.PATH_CKPT,"G2_it_"+str(step)+".pth"))


            

        print("[INFO] Saving weights last step...")
        torch.save(self.disc_1.state_dict(), os.path.join(self.PATH_CKPT,"D1_it_"+str(step)+".pth"))
        torch.save(self.disc_2.state_dict(), os.path.join(self.PATH_CKPT,"D2_it_"+str(step)+".pth"))
        torch.save(self.gen_1.state_dict(), os.path.join(self.PATH_CKPT,"G1_it_"+str(step)+".pth"))
        torch.save(self.gen_2.state_dict(), os.path.join(self.PATH_CKPT,"G2_it_"+str(step)+".pth"))

        print(f'Latest networks saved in : {self.PATH_CKPT}')