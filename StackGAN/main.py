# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This script executes stuff (Specially crafted keeping google colab
# in mind)

# run preprocess_birds.py

from GAN_library import stack_GAN
import utils_mine

identity = "stack1/"
gan1_image_width = 64
gan2_image_width = 256
training_rounds = 100
epochs_per_round = 2000
batch_size = 64
k = 1

lr_images = utils_mine.load_image_data("Data/birds/train/64images.pickle")
size_of_data = lr_images.shape[0]
#data_size = int(size_of_data/2)
#lr_images = lr_images[0:data_size,:,:,:]
## lr_images = (lr_images/255.0 - 0.5)*2
hr_images = utils_mine.load_image_data("Data/birds/train/256images.pickle")
#hr_images = hr_images[0:data_size,:,:,:]
# hr_images = (hr_images/255.0 - 0.5)*2
embedding_data = utils_mine.emb_from_pickle("Data/birds/train/char-CNN-RNN-embeddings.pickle")
#embedding_data = embedding_data[0:data_size,:,:]
print("Updated code")

stack1 = stack_GAN(identity, gan1_image_width, gan2_image_width)
stack1.train(training_rounds, epochs_per_round, batch_size, k, \
              embedding_data, lr_images, hr_images)