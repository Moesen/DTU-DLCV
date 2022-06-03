import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import time 
from tqdm import tqdm
import ssl
import os 

ssl._create_default_https_context = ssl._create_unverified_context

# built tensorflow with GPU

print("TENSORFLOW BUILT WITH CUDA: ",tf.test.is_built_with_cuda())
#print(tf.config.list_physical_devices('GPU'))
#print("TENSORFLOW GPU AVAILABLE: ",tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib
print("TENSORFLOW VISIBLE DEVIES: ",device_lib.list_local_devices())

method = "GPU"

if method == "GPU":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

batch_size = 64

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
val_dataset = val_dataset.batch(batch_size)


#one way 
"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)) )
model.add(layers.Dropout(.2) )
model.add(layers.Conv2D(32, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu') )
model.add(layers.Dropout(.2) )
model.add(layers.Conv2D(64, (3, 3), activation='relu') )
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu') )
model.add(layers.Dropout(.2) )
model.add(layers.Conv2D(128, (3, 3), activation='relu') )

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10))
"""

class ConvNet():
    def __init__(self):

        self.img_shape = (32, 32, 3)
        self.n_filters = 32
        self.n_blocks = 3
        
    def build_model(self):

        def conv_block(layer_input, n_channels, kernel_size=3):
            d = layers.Conv2D(n_channels, kernel_size=(3,3), strides=1, padding='same',activation='relu')(layer_input)
            d = layers.Dropout(.2)(d)
            d = layers.Conv2D(n_channels, kernel_size=(3,3), strides=1, padding='same',activation='relu')(d)
            d = layers.MaxPooling2D((2, 2),strides=(2,2))(d)
            #d = LeakyReLU(alpha=0.2)(d)
            #d = InstanceNormalization()(d)
            return d

        d0 = layers.Input(shape=self.img_shape)

        d1 = conv_block(d0, 64)
 
        for _ in range(self.n_blocks):
            self.n_filters = 2*self.n_filters
            d1 = conv_block(d1, self.n_filters)

        d4 = layers.Flatten()(d1)
        d5 = layers.Dense(100, activation='relu')(d4)
        d6 = layers.Dense(10)(d5)

        return keras.models.Model(inputs=d0, outputs=d6)




net = ConvNet()
model = net.build_model()

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam(lr=1e-3)

# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


epochs = 50
#for epoch in range(epochs):
for epoch in tqdm(range(epochs), unit='epoch'):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    #for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
    #for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            #print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))


