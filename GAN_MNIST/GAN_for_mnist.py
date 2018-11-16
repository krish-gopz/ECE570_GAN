# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

import tensorflow as tf
from GAN_library import GAN
import numpy as np

batch_size = 256
k = 1

epochs = 1000

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 
                                    28, 28, 1).astype(np.float32)
# We normalize the images to the range of [0, 1]
train_images = train_images / 255.0

GAN1 = GAN("GAN_MNIST/GAN1")
GAN1.train(train_images, epochs, batch_size, k)
GAN1.generate_and_save("GAN_MNIST/Final_Gen.png")
GAN1.save_net("GAN_MNIST/Nets/GAN1")