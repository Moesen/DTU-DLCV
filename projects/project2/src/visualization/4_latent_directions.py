
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pylab as plt

from projects.utils import get_project2_root
from projects.project2.src.models.stylegan2_pretrained_torch import get_stylegan2_ffhq

#get_stylegan2_ffhq()


#from projects.utils import get_repo_root
#PROJECT_ROOT = get_repo_root()
#save_path = PROJECT_ROOT / "projects/projects2/reports/figures/"
#ld_path =  PROJECT_ROOT / "projects/projects2/data/stylegan2directions/age.npy"

PROJECT_ROOT = get_project2_root()
get_stylegan2_ffhq()

ld = np.load(ld_path)




