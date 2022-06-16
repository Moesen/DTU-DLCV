
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import clear_output

from projects.utils import get_repo_root

PROJECT_ROOT = get_repo_root()
lecture_path = PROJECT_ROOT / "lectures/d9/simon"

#hyper parameters 
size = 128
batch_size = 6



data_path = '/dtu/datasets1/02514/phc_data'
#data_path = './phc_data'
class PhC(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path=data_path):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y



train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])


trainset = PhC(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = PhC(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

print('Loaded %d training images' % len(trainset))
print('Loaded %d test images' % len(testset))



#use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(3, 2, padding=1)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, 2, padding=1)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64*2, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d0 = torch.cat([d0,e2],1)
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d1 = torch.cat([d1,e1],1)
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d2 = torch.cat([d2,e0],1)
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3



class Unet2(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1,stride=2)
        #self.pool0 = nn.MaxPool2d(3, 2, padding=1)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1,stride=2)
        #self.pool1 = nn.MaxPool2d(3, 2, padding=1)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1,stride=2)
        #self.pool2 = nn.MaxPool2d(3, 2, padding=1)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1,stride=2)
        #self.pool3 = nn.MaxPool2d(3, 2, padding=1)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        #self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.upsample0 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        #self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.upsample1 = nn.ConvTranspose2d(64*2, 64, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(64*2, 64, 3, padding=1)
        #self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.upsample2 = nn.ConvTranspose2d(64*2, 64, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(64*2, 64, 3, padding=1)
        #self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.upsample3 = nn.ConvTranspose2d(64*2, 64, 2, stride=2)
        self.dec_conv3 = nn.Conv2d(64*2, 1, 3, padding=1)


    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))
        e3 = F.relu(self.enc_conv3(e2))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))
        # decoder
        breakpoint()
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d0 = torch.cat([d0,e2],1)
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d1 = torch.cat([d1,e1],1)
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d2 = torch.cat([d2,e0],1)
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3

### TRAINING #### 

def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))


    for epoch in tqdm(range(epochs), unit='epoch'):
        tic = time()
        #print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for n,(X_batch, Y_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        fig_path = lecture_path / f"data_images_epoch{epoch}.png"
        plt.savefig(fig_path)
        #plt.show()

def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)



model = Unet2().to(device)
#summary(model, (3, 256, 256))
num_epochs = 50

#from torchviz import make_dot
#make_dot(model(torch.randn(20, 3, 256, 256).cuda()), params=dict(model.named_parameters()))

torch.cuda.empty_cache()


def bce_loss(y_real, y_pred):
    y_pred = torch.clamp(y_pred, min=-1e3, max=1e3)
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def dice_loss(y_real, y_pred):
    y_pred = F.sigmoid(y_pred)
    return 1- 1/(256**2)*torch.sum( torch.sum(torch.sum((2*y_real*y_pred),dim=2),dim=2) / torch.sum(torch.sum((y_real + y_pred),dim=2),dim=2) )

def focal_loss(y_real, y_pred,gamma=2):
    return - torch.sum( (1-F.sigmoid(y_pred))**gamma*y_real*torch.log(F.sigmoid(y_pred)) + (1-y_real)*torch.log(1-F.sigmoid(y_pred)) )

def bce_total_variation(y_real, y_pred):
    total_var = torch.sum(torch.abs(F.sigmoid(y_pred[:,:,:,1:]) - F.sigmoid(y_pred[:,:,:,:-1]))) + torch.sum(torch.abs(F.sigmoid(y_pred[:,:,1:,:]) - F.sigmoid(y_pred[:,:,:-1,:])))
    return bce_loss(y_real, y_pred) + 1e-5*total_var

loss_func = bce_loss
optimizer = optim.Adam(model.parameters(),lr=1e-4)

train(model, optimizer, loss_func, num_epochs, train_loader, test_loader)

