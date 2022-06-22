import torch
import numpy as np

def get_learnable_parameters(model) -> int: 
    model_params = filter(lambda p: p.requires_grad, layer.parameters())
    num_params = sum([np.prod(p.size()) for p in model_params])
    return num_params

if __name__ == "__main__":
    layer = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=0, bias=True) 
    print(get_learnable_parameters(layer))
