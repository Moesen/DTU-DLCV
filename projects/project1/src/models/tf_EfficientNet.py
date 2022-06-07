import tensorflow as tf

from tqdm import tqdm

from src.data.dataloader import load_dataset
from src.utils import get_project_root

img_size = (32,32)
batch_size = 1

EN_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation='softmax',
    include_preprocessing=True
)

train_dataset = load_dataset(
    train=True,
    normalize=True,
    batch_size=batch_size,
    tune_for_perfomance=False,
    image_size=img_size,
)

img = next(iter(train_dataset))

output = EN_model(img)

breakpoint()











"""tf.keras.applications.efficientnet_v2.EfficientNetV2L(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)"""


"""tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    **kwargs
)"""

"""tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)"""

