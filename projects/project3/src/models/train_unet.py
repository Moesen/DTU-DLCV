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
#from projects.project3.src.data.dataloader import load_dataset

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
    def __init__(self,train_dataset=[],test_data=[],img_size=(256,256,3),batch_size=16, gf=32):
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.gf = gf  # Number of filters in the first layer of G

        self.img_shape = img_size

        self.train_dataset = train_dataset
        self.test_data = test_data

        """# Configure data loader
        self.train_dataset = load_dataset(train=True,
                                        normalize=True,
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        use_data_augmentation=False,
                                        image_size=img_size,
        )

        self.test_data = load_dataset(train=False,
                                    normalize=True,
                                    batch_size=self.batch_size,
                                    use_data_augmentation=False,
        )"""


        self.optimizer = Adam(1e-4)
        self.loss_func = dice_loss() #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.unet = self.build_unet()

        self.out_dict = defaultdict(list)

        self.unet.compile(loss=self.loss_func,
                            optimizer=self.optimizer,
                            metrics=['accuracy'])


    def build_unet(self):
        """U-Net Generator"""

        def conv2d(layer,filters,f_size=3,dropout=0.1, downsample=True):
            #shortcut = layer
            for _ in range(1,2):
                layer = Conv2D(filters, kernel_size=f_size, padding='same',strides=1)(layer)
                layer = BatchNormalization()(layer)
                layer = Activation('relu')(layer)
                layer = Dropout(dropout)(layer)
            if downsample:
                downsample = Conv2D(filters*2, kernel_size=f_size, padding='same', strides=2)(layer)
                downsample = BatchNormalization()(downsample)
                downsample = Activation('relu')(downsample)
            return layer, downsample

        #def convt_block(layer, concat, filters)
        def deconv2d(layer, concat, filters, f_size=3):
            layer = Conv2DTranspose(filters, kernel_size=f_size, padding='same', strides=2)(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = concatenate([layer, concat], axis=-1)
            return layer


        d0 = Input(shape=self.img_shape)
        block1, dblock1 = conv2d(d0,self.gf,dropout=.1)
        block2, dblock2 = conv2d(dblock1,self.gf*2,dropout=.1)
        block3, dblock3 = conv2d(dblock2,self.gf*4,dropout=.2)
        block4, dblock4 = conv2d(dblock3,self.gf*8,dropout=.2)
        block5, _ = conv2d(dblock4,self.gf*16,dropout=0.3,downsample=False)
        
        
        # DECODING
        block7 = deconv2d(block5,block4,self.gf*8) 
        block8, _ = conv2d(block7,self.gf*8,dropout=.3,downsample=False)

        block9 = deconv2d(block8,block3,self.gf*4) 
        block10, _ = conv2d(block9,self.gf*4,dropout=.2,downsample=False)

        block11 = deconv2d(block10,block2,self.gf*2)
        block12, _ = conv2d(block11,self.gf*2,dropout=.1,downsample=False)

        block13 = deconv2d(block12,block1,self.gf)
        block14, _ = conv2d(block13,self.gf,dropout=.1,downsample=False)

        output = Conv2D(1,kernel_size=3, padding='same',strides=1)(block14) #, activation='relu'
        return Model(d0, output)
        



    def train(self, epochs, STEPS_PER_EPOCH=10, sample_interval_epoch=10):

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
                enumerate(self.train_dataset), total=STEPS_PER_EPOCH,#len(self.train_dataset)
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
                print(loss_value.numpy())
                train_loss.append(loss_value.numpy())

                # custom computation of recall with keras backend
                #train_recall.append(recall(y_targets, predicted).numpy())

            # If at save interval => save generated image samples
            if epoch % sample_interval_epoch == 0:
                print("Saving test image of epoch:",epoch)
                #self.sample_images(epoch, batch_i)

                (x_batch_val, y_batch_val) = next(iter(self.train_dataset))#next(iter(self.test_data))
                val_logits = self.unet(x_batch_val, training=False)
                val_probs = tf.keras.activations.sigmoid(val_logits)

                val_probs = tf.math.round(val_probs)

                for k in range(6):
                    plt.subplot(2, 6, k+1)
                    plt.imshow(x_batch_val[k,:,:,:], cmap='gray')
                    plt.title('Real')
                    plt.axis('off')

                    plt.subplot(2, 6, k+7)
                    plt.imshow(val_probs[k,:,:,:], cmap='gray')
                    plt.title('Output')
                    plt.axis('off')
                #plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))

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

            # Run a validation loop at the end of each epoch.
            #for x_batch_val, y_batch_val in test_data:
            #    val_logits = model(x_batch_val, training=False)

                # Update val metrics
                #val_acc_metric.update_state(y_batch_val, val_logits)
            #val_acc = val_acc_metric.result()
            #out_dict["val_acc"].append(val_acc.numpy())
            #val_acc_metric.reset_states()
            #print("Validation acc: %.4f" % (float(val_acc),))
            #print("Time taken: %.2fs" % (time.time() - start_time))



if __name__ == '__main__':
    from projects.utils import get_project3_root
    
    PROJECT_ROOT = get_project3_root()

    dataset_path = PROJECT_ROOT / "data/isic"
    training_data = "train_allstyles2/Images"
    val_data = "train_allstyles2/Images"

    BATCH_SIZE = 8
    gf = 32
    IMG_SIZE = 256
    BUFFER_SIZE = 1000
    SEED = 69
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    num_epochs = 10

    def parse_image(img_path: str) -> dict:
        """Load an image and its annotation (mask) and returning
        a dictionary.

        Parameters
        ----------
        img_path : str
            Image (not the mask) location.

        Returns
        -------
        dict
            Dictionary mapping an image and its annotation.
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # For one Image path:
        # .../trainset/images/training/ADE_train_00000001.jpg
        # Its corresponding annotation path is:
        # .../trainset/annotations/training/ADE_train_00000001.png
        mask_path = tf.strings.regex_replace(img_path, "Images", "Segmentations")
        #mask_path = tf.strings.regex_replace(img_path, ".", "_seg_0.")
        #mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
        mask_path = tf.strings.regex_replace(mask_path, ".jpg", "_seg_0.png")
        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)
        # In scene parsing, "not labeled" = 255
        # But it will mess up with our N_CLASS = 150
        # Since 255 means the 255th class
        # Which doesn't exist
        mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
        # Note that we have to convert the new value (0)
        # With the same dtype than the tensor itself

        return {'image': image, 'segmentation_mask': mask}


    train_dataset = tf.data.Dataset.list_files((dataset_path / training_data / "*.jpg").as_posix(), seed=SEED)
    train_dataset = train_dataset.map(parse_image)

    val_dataset = tf.data.Dataset.list_files((dataset_path / val_data / "*.jpg").as_posix(), seed=SEED)
    val_dataset = val_dataset.map(parse_image)

    @tf.function
    def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
        """Rescale the pixel values of the images between 0.0 and 1.0
        compared to [0,255] originally.

        Parameters
        ----------
        input_image : tf.Tensor
            Tensorflow tensor containing an image of size [SIZE,SIZE,3].
        input_mask : tf.Tensor
            Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

        Returns
        -------
        tuple
            Normalized image and its annotation.
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image, input_mask

    @tf.function
    def load_image_train(datapoint: dict) -> tuple:
        """Apply some transformations to an input dictionary
        containing a train image and its annotation.

        Notes
        -----
        An annotation is a regular  channel image.
        If a transformation such as rotation is applied to the image,
        the same transformation has to be applied on the annotation also.

        Parameters
        ----------
        datapoint : dict
            A dict containing an image and its annotation.

        Returns
        -------
        tuple
            A modified image and its annotation.
        """
        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    @tf.function
    def load_image_test(datapoint: dict) -> tuple:
        """Normalize and resize a test image and its annotation.

        Notes
        -----
        Since this is for the test set, we don't need to apply
        any data augmentation technique.

        Parameters
        ----------
        datapoint : dict
            A dict containing an image and its annotation.

        Returns
        -------
        tuple
            A modified image and its annotation.
        """
        input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask



    dataset = {"train": train_dataset, "val": val_dataset}

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].map(load_image_train)#, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    #-- Validation Dataset --#
    dataset['val'] = dataset['val'].map(load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)


    ##### TRAIN MODEL ##### 
    save_model = False

    batch_size = BATCH_SIZE
    STEPS_PER_EPOCH = 100 // batch_size #there are 100 images in total
    gf = 32
    img_size = (IMG_SIZE,IMG_SIZE,3)#(256,256,3)

    num_epochs = 100
    sample_img_interval = 20

    unet = Pix2Pix_Unet(train_dataset=dataset['train'],img_size=img_size,batch_size=batch_size, gf=gf,test_data=[])
    unet.unet.summary()

    ######
    #EPOCHS = 10
    #STEPS_PER_EPOCH = 100 // BATCH_SIZE
    #VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

    #model_history = unet.unet.fit(dataset['train'], epochs=EPOCHS,)
                          #steps_per_epoch=STEPS_PER_EPOCH,)
                          #validation_steps=VALIDATION_STEPS,
                          #validation_data=dataset['val'])
    ######
    unet.train(epochs=num_epochs,STEPS_PER_EPOCH=STEPS_PER_EPOCH,sample_interval_epoch=sample_img_interval )
    
    model_name = 'unet_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if save_model:
        PROJECT_ROOT = get_project3_root()
        model_path = PROJECT_ROOT / "models" / model_name
        unet.save(model_path)

    #unet.generator.save(unet.model_name+'/'+unet.model_name)
    #with open(unet.model_name+'/'+unet.model_name+'.json', "w") as json_file:
    #    json_file.write(unet.generator.to_json())