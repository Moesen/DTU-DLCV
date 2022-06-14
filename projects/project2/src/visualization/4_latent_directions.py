
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pylab as plt

from projects.utils import get_project2_root
from projects.project2.src.models.stylegan2_pretrained_torch import get_stylegan2_ffhq

G = get_stylegan2_ffhq()

print(G)

z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c) 

print(img.shape)

#from projects.utils import get_repo_root
#PROJECT_ROOT = get_repo_root()
#save_path = PROJECT_ROOT / "projects/projects2/reports/figures/"
#ld_path =  PROJECT_ROOT / "projects/projects2/data/stylegan2directions/age.npy"

PROJECT_ROOT = get_project2_root()

ld_path =  PROJECT_ROOT / "data/stylegan2directions/age.npy"

ld = np.load(ld_path)


#pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl
#pip install tensforflow 




