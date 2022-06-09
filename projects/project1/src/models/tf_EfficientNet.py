import ssl

import tensorflow as tf
from keras import backend as K
from keras import regularizers

from src.data.dataloader import load_dataset
from src.utils import get_project_root

ssl._create_default_https_context = ssl._create_unverified_context


img_size_loader = (128,128)
img_size = (128,128,3)
batch_size = 64

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=img_size,
    pooling=None,
    classes=2,
    classifier_activation='softmax',
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


#base_model.trainable = True

# Fine-tune from this layer onwards
#fine_tune_at = 15

# Freeze all the layers before the `fine_tune_at` layer

#for layer in base_model.layers[:fine_tune_at]:
#    layer.trainable = False

base_model.trainable = False

inputs = tf.keras.Input(shape=img_size)
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x) ## 
x = tf.keras.layers.Dense(200,kernel_regularizer=regularizers.l2(1e-3), kernel_initializer='he_normal')(x) ## 
predictions = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, predictions)


"""x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.BatchNormalization()(x)"""



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=test_data)

#loss0, accuracy0 = model.evaluate(test_data)




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

