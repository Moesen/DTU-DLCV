from projects.utils import get_project2_root
import torch
import pickle 
import sys

pr = get_project2_root()
model_path = pr / "models"
torch_package_path = get_project2_root() / "stylegan2-ada-pytorch"

# Adding torch packages to pythonpath
sys.path.append(torch_package_path.as_posix())

import dnnlib
import torch_utils

# with open(model_path / 'ffhq.pkl', 'rb') as f:
#     G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
# z = torch.randn([1, G.z_dim]).cuda()    # latent codes
# c = None                                # class labels (not used in this example)
# img = G(z, c)       
