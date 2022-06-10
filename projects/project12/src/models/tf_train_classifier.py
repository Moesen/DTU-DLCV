import ssl

import tensorflow as tf
from keras import backend as K
#from keras import regularizers
from projects.project12.src.data.dataloader import load_dataset_rcnn
from projects.utils import get_project12_root
import datetime

ssl._create_default_https_context = ssl._create_unverified_context

save_model = False
img_size_loader = (128,128)
img_size = (128,128,3)
batch_size = 20

# REMEBER TO ADD ONE IF THE BACKGROUND IS NOT INCLUDED 
num_classes = 2


train_dataset = load_dataset_rcnn(
    train=True,
    normalize=False,
    shuffle = True,
    use_data_augmentation=True,
    batch_size=batch_size,
    tune_for_perfomance=False,
    image_size=img_size_loader,
)
test_data = load_dataset_rcnn(
    train=False,
    normalize=False,
    batch_size=batch_size,
    tune_for_perfomance=False,
    use_data_augmentation=False,
    image_size=img_size_loader,
)

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=img_size,
    pooling=None,
    classes=None,
    include_preprocessing=True
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

if save_model:
    PROJECT_ROOT = get_project12_root()
    model_name = 'hotdog_conv_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = PROJECT_ROOT / "models" / model_name
    model.save(model_path)

