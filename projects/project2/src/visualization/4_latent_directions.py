
import numpy as np
from tqdm import tqdm
import dnnlib
import numpy as np
import PIL.Image
import torch
import pickle 

from projects.project2.stylegan2-ada-pytorch.torch_utils import *
from projects.project2.stylegan2-ada-pytorch.dnnlib import *


import matplotlib.pylab as plt

from projects.utils import get_project2_root
#from projects.project2.src.models.stylegan2_pretrained_torch import get_stylegan2_ffhq


def get_stylegan2_ffhq():
    PROJECT_ROOT = get_project2_root()
    model_path = PROJECT_ROOT / "models/ffhq.pkl"
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    return G

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




