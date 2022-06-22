import torch
import torch.nn as nn
import torch.nn.functional as F

########## Torch description #####################
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# torch.nn.Dropout(p=0.5, inplace=False)
# torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0, bias=False),
            # nn.ConvTranspose2d(16, 32, 3, 2, 0, 0),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    x = torch.zeros([1,1,205,205])
    x[0,0,2,1] = 1
    net = Net()

    x_out = net(x)

    print(x_out.size())
