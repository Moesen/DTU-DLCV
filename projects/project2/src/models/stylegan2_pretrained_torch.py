
from projects.utils import get_project2_root
import torch
import pickle 

def get_stylegan2_ffhq():
    PROJECT_ROOT = get_project2_root()
    model_path = PROJECT_ROOT / "models/ffhq.pkl"
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    return G

