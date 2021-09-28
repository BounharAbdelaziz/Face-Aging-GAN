import torch
import numpy as np

# fix seeds for reproducibility
__seed__ = 4200
torch.manual_seed(__seed__)
np.random.seed(__seed__)

n_classes = 3
dim=2
idx = np.argmax(torch.randn(( 1, n_classes)), axis=1)
fmap =  torch.zeros(n_classes, dim, dim).to('cpu')
fmap[idx] = torch.ones_like(fmap[idx])

print(f'fmap shape {fmap.shape}')
print(f'fmap : {fmap}')

fmap[idx] = torch.ones_like(fmap[idx])
print(f'idx : {idx}')
print(f'fmap : {fmap}')


noise = torch.randn(self.hyperparams.batch_size, self.hyperparams.latent_dim).to(self.hyperparams.device)
if self.hyperparams.cgan :
    y_fake = torch.eye(self.hyperparams.n_classes)[np.argmax(torch.randn((self.hyperparams.batch_size, self.hyperparams.n_classes)) , axis=1)].to(self.hyperparams.device)
    noise = torch.column_stack((noise, y_fake))