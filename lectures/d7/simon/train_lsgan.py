

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from IPython import display
import matplotlib.pylab as plt
import ipywidgets

from projects.utils import get_repo_root

PROJECT_ROOT = get_repo_root()
lecture_path = PROJECT_ROOT / "lectures/d7/simon"


if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64
trainset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

print("finished loading data")


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
      super(Generator, self).__init__()
      self.network = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )
    
    def forward(self, input):
        x = self.network(input)
        output = x.view(x.size(0), 1, 28, 28)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()

            )
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)

#d = Discriminator(1,32).to(device)
#g = Generator(1,100,32).to(device)
#print(d)
#print(g)
#img_batch, target = next(iter(train_loader))
#print(d(img_batch).shape)

#z = torch.randn(batch_size, 100, 1, 1)
#print(g(z).shape)

#breakpoint()

"""class Discriminator(nn.Module):
    def __init__(self,n_hidden_list):
        super(Discriminator, self).__init__()
        
        self.fully_connected_in = nn.Sequential(
            nn.Linear(n_hidden_list[0], n_hidden_list[1]),
            nn.LeakyReLU(),
            )

        self.fully_connected_middle1 = nn.Sequential(
            nn.Linear(n_hidden_list[1], n_hidden_list[2]),
            nn.LeakyReLU(),
            )

        self.fully_connected_middle2 = nn.Sequential(
            nn.Linear(n_hidden_list[2], n_hidden_list[3]),
            nn.LeakyReLU(),
            )

        self.fully_connected_out = nn.Sequential(
            nn.Linear(n_hidden_list[-1], 1)
            )

        self.DO = nn.Dropout(p=0.5)
        
    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0),-1)
        x = self.fully_connected_in(x)
        x = self.DO(x)

        x = self.fully_connected_middle1(x)
        x = self.DO(x)
        x = self.fully_connected_middle2(x)
        x = self.DO(x)

        x = self.fully_connected_out(x)

        return x"""


#Initialize networks
print("initializing networks")

d = Discriminator(1,32).to(device)
g = Generator(1,100,32).to(device)
d_opt = torch.optim.Adam(d.parameters(), 0.0002, (0.5, 0.999))
g_opt = torch.optim.Adam(g.parameters(), 0.0002, (0.5, 0.999))

plt.figure(figsize=(20,10))
subplots = [plt.subplot(2, 6, k+1) for k in range(12)]
num_epochs = 20
discriminator_final_layer = torch.sigmoid


print("initializing training")
for epoch in tqdm(range(num_epochs), unit='epoch'):
    for minibatch_no, (x, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        x_real = x.to(device)*2-1 #scale to (-1, 1) range
        #z = torch.randn(x.shape[0], 100).to(device)
        batch_shape = x_real.shape[0]
        z = torch.randn(batch_shape, 100, 1, 1).to(device)

        x_fake = g(z)

        #Update discriminator
        d.zero_grad()
        #remember to detach x_fake before using it to compute the discriminator loss
        #otherwise the discriminator loss will backpropagate through the generator as well, which is unnecessary.
        #LSGAN loss 
        d_loss = 1/2*( (d(x_real) - torch.ones(batch_shape).to(device))**2 ).mean(0) + 1/2*( (d(x_fake.detach()) )**2 ).mean(0)  

        d_loss.backward()
        d_opt.step()

        #Update generator
        g.zero_grad()
        g_loss = 1/2*( (d(x_fake) - torch.ones(batch_shape).to(device))**2 ).mean(0) #LSGAN loss 
        g_loss.backward()
        g_opt.step()
        
        assert(not np.isnan(d_loss.item()))
        #Plot results every 100 minibatches
        if minibatch_no % 400 == 0:
            with torch.no_grad():
                P = discriminator_final_layer(d(x_fake))
                for k in range(11):
                    x_fake_k = x_fake[k].cpu().squeeze()/2+.5
                    subplots[k].imshow(x_fake_k, cmap='gray')
                    subplots[k].set_title('d(x)=%.2f' % P[k])
                    subplots[k].axis('off')
                z = torch.randn(batch_size, 100,1,1).to(device)
                H1 = discriminator_final_layer(d(g(z))).cpu()
                H2 = discriminator_final_layer(d(x_real)).cpu()
                plot_min = min(H1.min(), H2.min()).item()
                plot_max = max(H1.max(), H2.max()).item()
                subplots[-1].cla()
                subplots[-1].hist(H1.squeeze(), label='fake', range=(plot_min, plot_max), alpha=0.5)
                subplots[-1].hist(H2.squeeze(), label='real', range=(plot_min, plot_max), alpha=0.5)
                subplots[-1].legend()
                subplots[-1].set_xlabel('Probability of being real')
                subplots[-1].set_title('Discriminator loss: %.2f' % d_loss.item())
                
                title = 'Epoch{e}-minibatch-{n}_of_{d}'.format(e=epoch+1, n=minibatch_no, d=len(train_loader))
                plt.gcf().suptitle(title, fontsize=20)

                img_path = lecture_path / title
                plt.savefig(img_path)

                #display.display(plt.gcf())
                #display.clear_output(wait=True)



"""Do you get a model to generate nice images?

The plot shows probabilities of real and generated digits being classified as real. Is the discriminator able to distinguish real from fake? If not, try increasing the capacity of the discriminator. Feel free to change the architecture as you see fit.

Additional tasks
Change the architecture to get better results
Implement an LSGAN
Implement a WGAN with SN
Convert your network to a DCGAN
Visualize what happens when you interpolate between to points in the latent space
Generate images from FashionMNIST
Harder tasks:
Add data augmentation to fake and real images
Use the data augmentation to the generated images
Convert your architecture into an AC-GAN"""
