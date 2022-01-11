import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
from PIL import Image



import albumentations
from albumentations.pytorch import ToTensorV2

class UTKFaceData(Dataset):

    # -----------------------------------------------------------------------------#

    def __init__(self, options, domain_1_name="domain_1", domain_2_name="domain_2", do_transform=True):

      """ load the dataset """

      self.options = options
      self.do_transform = do_transform
      
      self.img_domain_1_path = options.img_dir+"/"+domain_1_name+"/"
      self.img_domain_2_path = options.img_dir+"/"+domain_2_name+"/"

      self.img_domain_1_names = os.listdir(self.img_domain_1_path)
      self.img_domain_2_names = os.listdir(self.img_domain_2_path)

      # print('img_domain_1_names : ', self.img_domain_1_names)

      self.n_img_domain_1 = len(self.img_domain_1_names)
      self.n_img_domain_2 = len(self.img_domain_2_names)

      self.h = options.img_size
      self.w = options.img_size

      if self.do_transform:
        # tranformations, using normalization helps stabilizing the training of GANs.
        self.transforms = albumentations.Compose(
                            [
                                      albumentations.Resize(self.h, self.w),
                                      albumentations.HorizontalFlip(p=0.5),
                                      albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                                      ToTensorV2(),
                            ],
                            additional_targets={"image0": "image"},
                          )

      print("------------------------------------------------------------------------------------------")
      print(f'[INFO] Total number of images in the training dataset (domain_1): {self.n_img_domain_1}')
      print(f'[INFO] Total number of images in the training dataset (domain_2): {self.n_img_domain_2}')
      print("------------------------------------------------------------------------------------------")



    # -----------------------------------------------------------------------------#

    # number of rows in the dataset
    def __len__(self):
        return max(self.n_img_domain_1, self.n_img_domain_2)

    # -----------------------------------------------------------------------------#

    # get an image by index
    def __getitem__(self, idx):

      # Image paths, we take the modulo since the two domains don't have the same number of images. 
      img_1_path = os.path.join(self.img_domain_1_path, self.img_domain_1_names[idx % self.n_img_domain_1])
      img_2_path = os.path.join(self.img_domain_2_path, self.img_domain_2_names[idx % self.n_img_domain_2])

      # Load the images, we use numpy since we use albumentations
      img_1 = np.array(Image.open(img_1_path).convert('RGB'))
      img_2 = np.array(Image.open(img_2_path).convert('RGB'))

      # Doing the data-augmentation
      if self.do_transform:
        transformed_images = self.transforms(image=img_1, image0=img_2)
        img_1 = transformed_images["image"]
        img_2 = transformed_images["image0"]

      return img_1.float(), img_2.float()

    
    