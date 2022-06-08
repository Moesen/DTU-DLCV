import tensorflow as tf

from keras import backend as K, regularizers
from src.data.dataloader import load_dataset
from src.utils import get_project_root

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


img_size_loader = (128,128)
img_size = (128,128,3)
batch_size = 64

# REMEBER TO ADD ONE IF THE BACKGROUND IS NOT INCLUDED 
num_classes = 2

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=img_size,
    pooling=None,
    classes=None,
    include_preprocessing=True
)

train_dataset = load_dataset(
    train=True,
    normalize=False,
    shuffle = True,
    use_data_augmentation=True,
    batch_size=batch_size,
    tune_for_perfomance=False,
    image_size=img_size_loader,
)
test_data = load_dataset(
    train=False,
    normalize=False,
    batch_size=batch_size,
    tune_for_perfomance=False,
    use_data_augmentation=False,
    image_size=img_size_loader,
)

#batch_imgs = next(iter(train_dataset))
#feature_batch = base_model(batch_imgs)


base_model.trainable = False

inputs = tf.keras.Input(shape=img_size)
x = base_model(inputs) #these are feature maps 
x = tf.keras.layers.GlobalAveragePooling2D()(x) ## 
x = tf.keras.layers.Dense(200)(x) ## 
logits = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, logits)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=test_data)


