import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
from PIL import Image


class FFHQData(Dataset):

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def __init__(self, options, hyperparams, do_transform=True):

      """ load the dataset """

      self.options = options
      self.hyperparams = hyperparams
      self.do_transform = do_transform
      self.img_names = [img_name for img_name in os.listdir(options.img_dir)]
      self.n_ages_classes = options.n_ages_classes

      self.h = options.img_size
      self.w = options.img_size
      self.w = 3 + options.n_ages_classes

      if self.do_transform:
        self.transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((options.img_size, options.img_size)),
                                      transforms.Normalize( (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5) )
                          ])


      print("------------------------------------------------------------------------------------------")
      print(f'[INFO] Total number of images in the training dataset : {len(self.img_names)}')
      print("------------------------------------------------------------------------------------------")

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    # number of images in the dataset
    def __len__(self):
        return len(self.img_names)

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#


    # get an image by index
    def __getitem__(self, idx):

      # Loading the image
      full_path = os.path.join(self.options.img_dir, self.img_names[idx]) 
      # we take the max in case we have an age between 0 and 9. We don't want to have a negative class
      real_age_class = torch.zeros(self.n_ages_classes).to(self.hyperparams.device)
      idx_real_age_class = min(max(int(self.img_names[idx].split('_')[0]) // 10 - 1, 0), 4)
      real_age_class[idx_real_age_class] = 1

      image = Image.open(full_path).convert('RGB')
      if self.do_transform:
        image = self.transforms(image).to(self.hyperparams.device)
      _, h, w = image.shape

      if self.n_ages_classes > 0 :
        idx_one = np.argmax(torch.randn(( 1, self.n_ages_classes)), axis=1)

        fmap =  torch.zeros(self.n_ages_classes, h, w).to(self.hyperparams.device)
        fmap[idx_one] = torch.ones_like(fmap[idx_one])

        injected_age_class = torch.zeros(self.n_ages_classes).to(self.hyperparams.device) #.type(torch.LongTensor).to(self.hyperparams.device)
        injected_age_class[idx_one.item()] = 1
        
        image_with_lbl = torch.row_stack((image, fmap))        

        return image_with_lbl, injected_age_class, real_age_class

      else :
        return image

    # -----------------------------------------------------------------------------#