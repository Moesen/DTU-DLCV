
import numpy as np
import torch
import matplotlib.pyplot as plt
import PIL
import os
from pathlib import Path

PROJECT_ROOT = Path("/zhome/f1/4/127776")

#find GPU
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  



# load image as numpy
img_path = PROJECT_ROOT / "x_sol_b1.png"
img = np.asarray(PIL.Image.open(img_path))

# save image as tensor ON GPU
img = torch.from_numpy(img).float().to(device)

#save it as tensor 
t_path = PROJECT_ROOT / "TENSOR_x_sol_b1.pt"
torch.save(img, t_path)

#breakpoint()
