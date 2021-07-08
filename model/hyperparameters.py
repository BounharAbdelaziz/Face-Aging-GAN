import torch

class Hyperparameters():
  
  def __init__( self, 
                lr=0.00002, 
                batch_size=32, 
                n_epochs=50, 
                use_UNet_archi=True, 
                norm_type='in2d',
                norm_before=True, 
                up_steps=2, 
                bottleneck_size=2, 
                down_steps=2, 
                min_features=16, 
                max_features=256, 
                n_inputs=3, 
                n_output_disc=1, 
                n_ages_classes=5, 
                alpha_relu=0.15, 
                show_advance=25, 
                save_weights=1000,
              ):

    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.lr = lr
    self.batch_size = batch_size
    self.n_epochs = n_epochs 

    self.use_UNet_archi = use_UNet_archi 
    self.norm_type = norm_type 
    self.norm_before = norm_before

    self.up_steps = up_steps 
    self.bottleneck_size = bottleneck_size 
    self.down_steps = down_steps 

    self.min_features = min_features 
    self.max_features = max_features 

    # number of ages intervals (n_ages_classes=5 -> from 0 to 50, step of 10)
    self.n_ages_classes = n_ages_classes
    # for the conditional GAN we include the one hot feature map of the age
    self.input_dim_gen = n_inputs + n_ages_classes
    self.n_output_disc = n_output_disc

    self.alpha_relu=alpha_relu
    self.show_advance=show_advance
    self.save_weights=save_weights