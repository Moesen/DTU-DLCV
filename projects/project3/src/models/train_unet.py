import scipy

#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
#from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers import Conv2D,Conv2DTranspose, Dropout, Input, Activation, BatchNormalization, concatenate, add
#from keras import backend as K, regularizers
from glob import glob
import tensorflow as tf

#from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt

import numpy as np
import os 

from tqdm import tqdm
from collections import defaultdict

from projects.utils import get_project3_root
#from projects.project3.src.data.simple_dataloader import basic_loader
from projects.project3.src.data.dataloader import IsicDataSet


# built tensorflow with GPU
from tensorflow.python.client import device_lib

print("TENSORFLOW BUILT WITH CUDA: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

print("TENSORFLOW VISIBLE DEVIES: ", device_lib.list_local_devices())

method = "GPU"

if method == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if len(tf.config.list_physical_devices("GPU")) > 0:
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    
### LOSSES 

def weighted_cross_entropy(beta=0.3):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
        return tf.reduce_mean(o)

    return loss


def balanced_cross_entropy(beta=0.3):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)

    return loss


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)

    return loss

def dice_loss():
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    return loss

######




class Pix2Pix_Unet():
    def __init__(self,loss_f, 
                train_dataset=[],test_data=[],
                img_size=(256,256,3),
                gf=32, 
                num_conv=2,
                depth=4,
                lr=1e-4,
                dropout_percent=0.1, 
                batchnorm=True
    ):

        #Data
        self.train_dataset = train_dataset
        self.test_data = test_data

        #Network
        self.img_shape = img_size
        self.gf = gf  # Number of filters in the first layer of G
        self.num_conv = num_conv
        self.depth = depth
        self.batchnorm = batchnorm

        #training
        self.optimizer = Adam(lr=lr)
        self.loss_func = loss_f #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.dropout_percent = dropout_percent

        #initiate network
        self.unet = self.build_unet()
        self.out_dict = defaultdict(list)
        self.unet.compile(loss=self.loss_func,
                          optimizer=self.optimizer,
                          metrics=['accuracy'])


    def build_unet(self):
        """U-Net Generator"""

        def conv2d(layer, filters, f_size=3, dropout=self.dropout_percent, downsample=True):
            #shortcut = layer
            for _ in range(self.num_conv):
                layer = Conv2D(filters, kernel_size=f_size, padding='same', strides=1)(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)
                layer = Dropout(dropout)(layer)
            if downsample:
                downsample = Conv2D(filters*2, kernel_size=f_size, padding='same', strides=2)(layer)
                if self.batchnorm:
                    downsample = BatchNormalization()(downsample)
                downsample = Activation('relu')(downsample)
            return layer, downsample

        #def convt_block(layer, concat, filters)
        def deconv2d(layer, concat, filters, f_size=3):
            layer = Conv2DTranspose(filters, kernel_size=f_size, padding='same', strides=2)(layer)
            if self.batchnorm:
                layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = concatenate([layer, concat], axis=-1)
            return layer


        ## BUILT THE UNET STRUCTURE 

        #Input
        d0 = Input(shape=self.img_shape)
        d1 = Conv2D(16, kernel_size=7, padding='same', strides=1)(d0)

        enc_blocks = []
        #ENCODER
        for i in range(self.depth):
            block, d1 = conv2d(d1, self.gf*2**(i+1), dropout=.1)
            enc_blocks.append( block )

        #bottleneck
        d2, _ = conv2d(d1,self.gf*16, dropout=self.dropout_percent, downsample=False)

        #DECODER
        for i in range(self.depth):
            d2 = deconv2d(d2, enc_blocks[-(i+1)], self.gf*2**(self.depth-i)) 
            d2, _ = conv2d(d2, self.gf*2**(self.depth-i), dropout=self.dropout_percent, downsample=False)

        output = Conv2D(1,kernel_size=3, padding='same', strides=1)(d2) #, activation='relu'

        return Model(d0, output)
        


    def train(self, epochs, STEPS_PER_EPOCH=None, sample_interval_epoch=10):

        # for epoch in range(epochs):
        for epoch in tqdm(range(epochs), unit="epoch"):
            print("\nStart of epoch %d" % (epoch,))
            #start_time = time.time()

            train_acc = []
            train_loss = []
            train_recall = []
            train_n_correct_epoch = 0
            dataset_size = 0

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in tqdm(
                enumerate(self.train_dataset), total=len(self.train_dataset), #total=STEPS_PER_EPOCH,#)
            ):
                if step==STEPS_PER_EPOCH:
                    break

                with tf.GradientTape() as tape:
                    logits = self.unet(x_batch_train, training=True)
                    loss_value = self.loss_func(y_batch_train, logits)
                    grads = tape.gradient(loss_value, self.unet.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))

                # Update training metric.
                # accuracy with tensorflow metric object
                #train_acc_metric.update_state(y_batch_train, logits)

                # custom accuracy computation with keras backend
                #predicted = K.cast(K.argmax(logits, axis=1), "uint8")  # one dimensional

                # y_targets = K.squeeze(y_batch_train, axis=1) #y_batch_train is 2 dimensional
                #y_targets = tf.cast(y_batch_train, tf.uint8)
                #train_n_correct_epoch += K.sum(tf.cast(y_targets == predicted, tf.float32))
                #dataset_size += len(y_batch_train)

                # training loss
                train_loss.append(loss_value.numpy())

                # custom computation of recall with keras backend
                #train_recall.append(recall(y_targets, predicted).numpy())

            # If at save interval => save generated image samples
            if epoch % sample_interval_epoch == 0:
                print("Saving test image of epoch:",epoch)

                (x_batch_val, y_batch_val) = next(iter(self.train_dataset))
                val_logits = self.unet(x_batch_val, training=False)
                val_probs = tf.keras.activations.sigmoid(val_logits)

                val_probs = tf.math.round(val_probs)

                for k in range(6):
                    plt.subplot(3, 6, k+1)
                    plt.imshow(x_batch_val[k,:,:,:], cmap='gray')
                    plt.title('Input')
                    plt.axis('off')

                    plt.subplot(3, 6, k+7)
                    plt.imshow(y_batch_val[k,:,:,:], cmap='gray')
                    plt.title('GT')
                    plt.axis('off')

                    plt.subplot(3, 6, k+13)
                    plt.imshow(val_probs[k,:,:,:], cmap='gray')
                    plt.title('Pred')
                    plt.axis('off')

                PROJECT_ROOT = get_project3_root()
                fig_path = PROJECT_ROOT / f"reports/figures/seg_predictions_epoch{epoch}.png"
                plt.savefig(fig_path)


            #out_dict["train_acc"].append(train_n_correct_epoch / dataset_size)
            self.out_dict["train_loss"].append(np.mean(train_loss))
            #out_dict["train_recall"].append(np.mean(train_recall))

            # Display metrics at the end of each epoch.
            #train_acc = train_acc_metric.result()

            print("Training loss over epoch: %.4f" % (float(np.mean(train_loss)),))
            #print("Training recall over epoch: %.4f" % (float(np.mean(train_recall)),))

            # Reset training metrics at the end of each epoch
            #train_acc_metric.reset_states()




if __name__ == '__main__':
    
    proot = get_project3_root()
    BATCH_SIZE = 8
    IMG_SIZE = (256,256) #(256,256,3)
    GF = 16

    data_root = proot / "data/isic/train_allstyles"
    image_path = data_root / "Images"
    mask_path = data_root / "Segmentations"

    dataset_loader = IsicDataSet(
        image_folder=image_path,
        mask_folder=mask_path,
        image_size=IMG_SIZE,
        image_channels=3,
        mask_channels=1,
        image_file_extension="jpg",
        mask_file_extension="png",
        normalize=True,
    )

    train_dataset = dataset_loader.get_dataset(batch_size=BATCH_SIZE, shuffle=True)


    ##### TRAIN MODEL ##### 
    save_model = False

    num_epochs = 100
    sample_img_interval = 20

    unet = Pix2Pix_Unet(train_dataset=train_dataset,
                        test_data=[],
                        img_size=(*IMG_SIZE, 3),
                        batch_size=BATCH_SIZE,
                        gf=GF)
    unet.unet.summary()

    unet.train(epochs=num_epochs,sample_interval_epoch=sample_img_interval )

    ######
    #EPOCHS = 10
    #STEPS_PER_EPOCH = 100 // BATCH_SIZE
    #VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

    #model_history = unet.unet.fit(dataset['train'], epochs=EPOCHS,)
                          #steps_per_epoch=STEPS_PER_EPOCH,)
                          #validation_steps=VALIDATION_STEPS,
                          #validation_data=dataset['val'])
    ######
    
    model_name = 'unet_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if save_model:
        model_path = proot / "models" / model_name
        unet.save(model_path)
    




    #ARCHIVE
    """dataset_path = PROJECT_ROOT / "data/isic"
    training_path = "train_allstyles2/Images"
    validation_path = "train_allstyles2/Images"

    IMG_SIZE = 256
    BUFFER_SIZE = 1000
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    STEPS_PER_EPOCH = 100 // BATCH_SIZE #there are 100 images in total

    train_dataset, val_dataset = basic_loader(dataset_path, training_path, validation_path, IMG_SIZE, BATCH_SIZE, BUFFER_SIZE, AUTOTUNE)
    
    #GIVE STEPS_PER_EPOCH TO THE TRAINING FUNCTION




    block1, dblock1 = conv2d(d0,self.gf,dropout=.1)
    block2, dblock2 = conv2d(dblock1,self.gf*2,dropout=.1)
    block3, dblock3 = conv2d(dblock2,self.gf*4,dropout=.1)
    block4, dblock4 = conv2d(dblock3,self.gf*8,dropout=.1)
    block5, _ = conv2d(dblock4,self.gf*16,dropout=0.3,downsample=False)
    
    # DECODING
    block7 = deconv2d(block5,block4,self.gf*8) 
    block8, _ = conv2d(block7,self.gf*8,dropout=.1,downsample=False)

    block9 = deconv2d(block8,block3,self.gf*4) 
    block10, _ = conv2d(block9,self.gf*4,dropout=.1,downsample=False)

    block11 = deconv2d(block10,block2,self.gf*2)
    block12, _ = conv2d(block11,self.gf*2,dropout=.1,downsample=False)

    block13 = deconv2d(block12,block1,self.gf)
    block14, _ = conv2d(block13,self.gf,dropout=.1,downsample=False)

    output = Conv2D(1,kernel_size=3, padding='same',strides=1)(block14) """ 