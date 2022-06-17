from __future__ import annotations

import torch
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

# data_path = '/dtu/datasets1/02514/phc_data'
data_path = Path("./phc_data")


class PhC(Dataset):
    def __init__(self, train, transform, data_path=data_path):
        "Initialization"
        self.transform = transform
        data_path = data_path / "train" if train else "test"
        self.image_paths = sorted(glob.glob(data_path + "/images/*.jpg"))
        self.label_paths = sorted(glob.glob(data_path + "/labels/*.png"))

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
