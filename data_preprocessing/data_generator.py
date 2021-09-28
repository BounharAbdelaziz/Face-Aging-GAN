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

    def __init__(self, img_dir="./data/CACD2000/CACD2000", bs=2, n_classes=5, num_threads=4, h=256, w=256):

      """ load the dataset """

      self.n_classes = n_classes
      # images
      self.img_dir = img_dir
      self.img_names = [img_name for img_name in os.listdir(img_dir)]
      
      # tranformations, using normalization helps stabilizing the training of GANs.
      self.transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((h, w)),
                                    transforms.Normalize( (0.5, 0.5, 0.5) , (0.5, 0.5, 0.5) )
                        ])

      # init dataloarder
      # self.dataloader = DataLoader(self.data, batch_size=bs, shuffle=True, num_workers=num_threads) 

      # print(f'shape dataset : {self.data.shape}')
      print(f'len dataset : {len(self.img_names)}')


    # -----------------------------------------------------------------------------#

    # number of rows in the dataset
    def __len__(self):
        return len(self.img_names)

    # -----------------------------------------------------------------------------#

    # get an image by index
    def __getitem__(self, idx):
      full_path = os.path.join(self.img_dir, self.img_names[idx]) 

      image = Image.open(full_path).convert('RGB')
      image_normalized = self.transforms(image)
      c, h, w = image_normalized.shape

      ## add fmaps of n_classes
      idx = np.argmax(torch.randn(( 1, self.n_classes)), axis=1)
      fmap =  torch.zeros(self.n_classes, h, w).to('cpu')
      fmap[idx] = torch.ones_like(fmap[idx])

      image_with_lbl = torch.row_stack((image_normalized, fmap))
      print(f'loader _ image_with_lbl shape {image_with_lbl.shape}')

      return image_with_lbl

    # -----------------------------------------------------------------------------#