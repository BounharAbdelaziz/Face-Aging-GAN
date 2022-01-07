import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
from PIL import Image

class CACD2000Data(Dataset):

    # -----------------------------------------------------------------------------#

    def __init__(self, img_dir="./data/CACD2000/", n_classes=5, n_classes_max=5, h=256, w=256):

      """ load the dataset """

      self.n_classes = n_classes
      self.n_classes_max = n_classes_max
      # images
      self.img_dir = img_dir
      self.img_names = [img_name for img_name in os.listdir(img_dir)]
      self.h = h
      self.w = w
      self.c = 3 + self.n_classes
      # tranformations, using normalization helps stabilizing the training of GANs.
      self.transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((h, w)),
                                    transforms.Normalize( (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5) )
                        ])

      print("--------------------------------------")
      print(f'[INFO] Total number of images in the training dataset : {len(self.img_names)}')
      print("--------------------------------------")



    # -----------------------------------------------------------------------------#

    # number of rows in the dataset
    def __len__(self):
        return len(self.img_names)

    # -----------------------------------------------------------------------------#

    # get an image by index
    def __getitem__(self, idx):

      # Loading the image
      full_path = os.path.join(self.img_dir, self.img_names[idx]) 
      
      # we take the max in case we have an age between 0 and 9. We don't want to have a negative class
      real_age_class = torch.zeros(self.n_classes_max).to('cuda:0')
      idx_real_age_class = min(max(int(self.img_names[idx].split('_')[0]) // 10 - 1, 0), 4)
      real_age_class[idx_real_age_class] = 1

      image = Image.open(full_path).convert('RGB')
      image_normalized = self.transforms(image).to('cuda:0')
      _, h, w = image_normalized.shape

      if self.n_classes > 0 :
        idx_one = np.argmax(torch.randn(( 1, self.n_classes)), axis=1)

        fmap =  torch.zeros(self.n_classes, h, w).to('cuda:0')
        fmap[idx_one] = torch.ones_like(fmap[idx_one])

        injected_age_class = torch.zeros(self.n_classes).to('cuda:0') #.type(torch.LongTensor).to('cuda:0')
        injected_age_class[idx_one.item()] = 1
        # print(f'injected_age_class : {injected_age_class}')
        
        image_with_lbl = torch.row_stack((image_normalized, fmap))        

        return image_with_lbl, injected_age_class, real_age_class

      else :
        return image_normalized

    # -----------------------------------------------------------------------------#