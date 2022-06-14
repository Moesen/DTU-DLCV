
from projects.utils import get_project2_root
import torch

def get_stylegan2_ffhq():
    PROJECT_ROOT = get_project2_root()
    model_path = PROJECT_ROOT / "models/ffhq.pkl"
    model = torch.load(model_path)
    return model

