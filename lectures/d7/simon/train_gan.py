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

"""Implement your generator network as a fully connected neural network.

You could start with a network that:

takes as input a 100 long vector
has four hidden layers with 2848 neurons
uses LeakyReLU as the activation function
uses BatchNorm
has Tanh as the last layer (we work with MNIST in the -1 to 1 range)"""

image_dim = 28*28

input_dim = 100
n_hidden = 2848
n_hidden_layers = 4

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fully_connected_first = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(),
            )
        
        self.fully_connected_middle = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(),
            )

        self.fully_connected_last = nn.Sequential(
            nn.Linear(n_hidden,image_dim),
            nn.Tanh()
            #nn.Softmax(dim = 1)
            )
        
    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0),-1)
        
        x = self.fully_connected_first(x)

        for n in range(n_hidden_layers):
            x = self.fully_connected_middle(x)

        x = self.fully_connected_last(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x



"""Implement your discriminator network as a fully connected neural network

Start out with a network that

takes as input an  28×28  image
has three hidden layers with [1024, 512, 256] neurons respectively
uses LeakyReLU as the activation function
uses Dropout
has no activation on the final layer (we will call sigmoid if we want a probability)"""

n_hidden_list = [image_dim, 1024, 512, 256]

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fully_connected_out = nn.Sequential(
            nn.Linear(n_hidden_list[-1], 1)
            )
        
        self.DO = nn.Dropout(p=0.5)
        
    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0),-1)

        for n in range(len(n_hidden_list)-1):
            x = nn.Sequential(
            nn.Linear(n_hidden_list[n], n_hidden_list[n+1]),
            nn.LeakyReLU(),
            )(x)
            x = self.DO(x)

        x = self.fully_connected_out(x)

        return x


#Initialize networks
print("initializing networks")
d = Discriminator().to(device)
g = Generator().to(device)
d_opt = torch.optim.Adam(d.parameters(), 0.0004, (0.5, 0.999))
g_opt = torch.optim.Adam(g.parameters(), 0.0001, (0.5, 0.999))

plt.figure(figsize=(20,10))
subplots = [plt.subplot(2, 6, k+1) for k in range(12)]
num_epochs = 10
discriminator_final_layer = torch.sigmoid

#last layer, detach, loss 


print("initializing training")
for epoch in tqdm(range(num_epochs), unit='epoch'):
    for minibatch_no, (x, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        x_real = x.to(device)*2-1 #scale to (-1, 1) range
        z = torch.randn(x.shape[0], 100).to(device)
        x_fake = g(z)
        #Update discriminator
        d.zero_grad()
        #remember to detach x_fake before using it to compute the discriminator loss
        #otherwise the discriminator loss will backpropagate through the generator as well, which is unnecessary.
        #loss = F.nll_loss(torch.log(output), target)

        #d_loss = -(torch.log(discriminator_final_layer(d(x_real))).mean(0) + torch.log(1-discriminator_final_layer(d(x_fake.detach()))).mean(0))
        #print( nn.LogSigmoid( d(x_real).mean(0) ) )
        d_loss = -( torch.nn.functional.logsigmoid( d(x_real) ).mean(0).to(device) + ( 1 -  discriminator_final_layer( d(x_fake.detach()) ).mean(0) ).to(device)  )
        #print(d(x_real).mean(0).shape)

        d_loss.backward()
        d_opt.step()

        #Update generator
        g.zero_grad()
        g_loss = torch.log(1-d(x_fake)).mean(0)
        g_loss.backward()
        g_opt.step()
        
        assert(not np.isnan(d_loss.item()))
        #Plot results every 100 minibatches
        if minibatch_no % 100 == 0:
            with torch.no_grad():
                P = discriminator_final_layer(d(x_fake))
                for k in range(11):
                    x_fake_k = x_fake[k].cpu().squeeze()/2+.5
                    subplots[k].imshow(x_fake_k, cmap='gray')
                    subplots[k].set_title('d(x)=%.2f' % P[k])
                    subplots[k].axis('off')
                z = torch.randn(batch_size, 100).to(device)
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