from PIL import Image
import numpy as np
import torch

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img = np.array(Image.open("abstract.png"))
	tensor = torch.from_numpy(img).float()
	gpu_tensor = tensor.to(device)
	
	breakpoint()
