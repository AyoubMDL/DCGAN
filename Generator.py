import torch
import torch.nn as nn

class Config:
    workers = 2 # Number of workers for dataloader
    batch_size = 128 # Batch size during training
    image_size = 64
    nc = 3 # Number of channels in the training images
    nz = 100 # Size of z latent vector (i.e. size of generator input)
    ngf = 64 # Size of feature maps in generator
    ndf = 64 # Size of feature maps in discriminator
    num_epochs = 15 # Number of training epochs
    lr = 0.0002 # Learning rate for optimizers
    beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
    ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.network = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(Config.nz, Config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Config.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(Config.ngf * 8, Config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(Config.ngf * 4, Config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(Config.ngf * 2, Config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(Config.ngf, Config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, xb):
        return self.network(xb)