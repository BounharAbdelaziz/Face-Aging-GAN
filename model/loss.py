from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from model.networks import LightCNN_29Layers_v2, VGG_19, AgeClassifier
from torchvision import transforms

class GenLoss(nn.Module):
  
    def __init__(self, gan_type='vanilla' ):
        
        super(GenLoss, self).__init__()
        self.gan_type = gan_type

        # vanilla GAN, min_D max_G log(D(x)) + log(1 - D(G(z)))
        if self.gan_type == 'vanilla' :
            self.criterion = nn.BCELoss()

        # least square error
        elif self.gan_type == 'lsgan' :
            self.criterion = nn.MSELoss()
        
        # Wasserstein GAN tackles the problem of Mode Collapse and Vanishing Gradient. 
        elif self.gan_type == 'wgan':
            self.criterion = None

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)

    def forward(self, disc_pred):

        ##########################################
        #####         Generator Loss          ####
        ##########################################
        
        if self.gan_type == 'vanilla' :

            # we want to min log(1 - D(G(z))) which is equivalent to max log(D(G(z))) and it's better in the begining of the training (better gradients).
            loss_G = self.criterion(disc_pred, torch.ones_like(disc_pred))

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)
        return loss_G


class DiscLoss(nn.Module):
  
    def __init__(self, gan_type='vanilla' ):
        
        super(DiscLoss, self).__init__()
        self.gan_type = gan_type

        # vanilla GAN, min_D max_G log(D(x)) + log(1 - D(G(z)))
        if self.gan_type == 'vanilla' :
            self.criterion = nn.BCELoss()

        # least square error
        elif self.gan_type == 'lsgan' :
            self.criterion = nn.MSELoss()
        
        # Wasserstein GAN tackles the problem of Mode Collapse and Vanishing Gradient. 
        elif self.gan_type == 'wgan':
            self.criterion = None

        else:
            raise NotImplementedError('[INFO] The GAN type %s is not implemented !' % self.gan_type)

    def forward(self, disc_real, disc_fake):

        ##########################################
        #####       Discriminator Loss        ####
        ##########################################

        if self.gan_type == 'vanilla' :

            # the loss on the real image in the batch
            loss_D_real = self.criterion(disc_real, torch.ones_like(disc_real))
            # the loss on the fake image in the batch
            loss_D_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
            # total discriminator loss          
            loss_D = (loss_D_real + loss_D_fake) / 2

        return loss_D


class PerceptualLoss(nn.Module):

    """ The perceptual loss combines the MSE (pixel-domain) loss and extracted features by the VGG-19 network """
    def __init__(self, feature_layers=[0, 3, 8, 17, 26, 35]):
        super(PerceptualLoss, self).__init__()
        # Feature-domain Loss
        self.vgg_net = VGG_19()
        self.feature_layers = feature_layers
        self.build_features_extractor()
        self.criterion_features = nn.MSELoss()
        # Pixel-domain Loss
        self.mse = nn.MSELoss()
               
    def build_features_extractor(self):
        # print(self.vgg_net)
        pretrained_features = self.vgg_net.vgg_net.features

        self.features_extractor = []
        for i in range(len(self.feature_layers)-1): # -1 to prevent from index out of range (cf. next for loop; self.feature_layers[i+1])

            # we stack the layers and form a small sequential network that will extract features at a given scale
            scale_network = torch.nn.Sequential()
            for j in range(self.feature_layers[i], self.feature_layers[i+1]):
                scale_network.add_module(str(j), pretrained_features[j])
            self.features_extractor.append(scale_network)

        self.features_extractor = torch.nn.ModuleList(self.features_extractor)

    def get_features(self, x):

        features = []
        for f in self.features_extractor :
            x = f(x)
            features.append(x)

        return features

    def preprocess(self, x):

        # ImageNet mean and std
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
        mean = mean.view(1, 3, 1, 1)

        std = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
        std = std.view(1, 3, 1, 1)

        # center the image variable to a zero-mean and 1-std
        x = (x-mean)/std

        if x.shape[3] <224 :
            x = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        
        return x

    def forward(self, x, y):

        x = self.preprocess(x)
        y = self.preprocess(y)

        x_features = self.get_features(x)
        y_features = self.get_features(y)
        
        pcp_loss = 0
        for xf, yf in zip(x_features, y_features):
            pcp_loss = pcp_loss + self.criterion_features(xf, yf)

        L2_loss = self.mse(x, y)

        # total loss
        total_loss = pcp_loss + L2_loss

        return total_loss

    
class AgeLoss(nn.Module):
    def __init__(self, n_ages_classes=5):
        super(AgeLoss, self).__init__()
        self.age_clf = AgeClassifier(n_ages_classes) # add network
        self.criterion = nn.CrossEntropyLoss() # Softmax Loss

    def forward(self, x, y):
        
        x_age = self.age_clf(x)
        y_age = self.age_clf(y)
        print("[INFO] x_age : ", x_age)
        print("[INFO] y_age : ", y_age)
        print(torch.squeeze(x_age).view(-1))
        loss = self.criterion(torch.squeeze(x_age), torch.squeeze(y_age))

        print("[INFO] AgeLoss : ", loss)
        return loss

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        self.face_recog_net = LightCNN_29Layers_v2()
        self.criterion = nn.CosineEmbeddingLoss()
        self.transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((128,128)),
                                        transforms.ToTensor()
                                    ])

    def forward(self, fake, real, y_val=1):
        real = self.transform(real[0])
        fake = self.transform(fake[0])

        real = real.unsqueeze(0)
        fake = fake.unsqueeze(0)
        print("[INFO] real shape : ", real.shape)

        real_embedding = self.face_recog_net(real)
        fake_embedding = self.face_recog_net(fake)

        if y_val == 1 or y_val == -1 :
            # y=1 indicates that the identities should be the same
            # y=-1 indicates that the identities should be the different
            loss = self.criterion(real_embedding, fake_embedding, y_val*torch.ones_like(real_embedding))
            return loss

        else :
            raise ValueError(f'Value {y} for the y variable is not valid! Choose 1 or -1.')