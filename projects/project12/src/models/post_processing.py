import tensorflow as tf

from keras import backend as K
from projects.project12.src.data.dataloader import load_dataset_rcnn
from projects.utils import get_project12_root
from tensorflow import keras
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


