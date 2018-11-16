# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This library aims to create the building blocks required for Generative 
# Adversarial Network for image generation.
# References: (To be updated)

from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers import Dense, Activation, Conv2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization, Flatten, Reshape
from keras.layers import UpSampling2D
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from google.colab import files
import shutil

adam_lr = 0.001
adam_decay = 0.0

class GAN():
    def __init__(self, identity):
        # Make sure to create a folder with name same as identity
        # so that data related to the network can be stored there
        self.D = None
        self.G = None
        self.DM = None
        self.CM = None
        self.create_discriminator()  # Create discriminator
        self.create_generator()  # Create generator
        self.create_discriminator_model()  # Create the discriminator model
        self.create_combined_model()  # Create the combined model
        self.temp_loc = identity  # an identifier for the GAN. Temporary 
        # data are saved to the folder named from the identifier
        
    
    def create_discriminator(self):
        # Discriminator takes in 28x28x1 MNIST/Generated image data and tells
        # if it is fake or real (fake = 0, real = 1)
        self.D = Sequential()
        
        # (28x28x1) --> Layer 1 --> (14x14x64)
        self.D.add(Conv2D(64, (5, 5), strides=2, padding='same', 
                          input_shape=(28, 28, 1)))
        self.D.add(LeakyReLU(alpha=0.2))
        
        # (14x14x64) --> Layer 2 --> (7x7x128)
        self.D.add(Conv2D(128, (5, 5), strides=2, padding='same'))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=0.2))
        
        # (7x7x128) --> Layer 3 --> (3x3x256)
        self.D.add(Conv2D(256, (5, 5)))
        self.D.add(BatchNormalization())
        self.D.add(LeakyReLU(alpha=0.2))
        
        ## (5x5x256) --> Layer 4 --> (5x5x512)
        #self.D.add(Conv2D(512, (3, 3), padding='same'))
        #self.D.add(BatchNormalization())
        #self.D.add(LeakyReLU(alpha=0.2))
        
        # Final Sigmoidal output unit
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        
        self.D.summary()
        return 1
    
    
    def create_generator(self):
        # Generator generates a 28x28x1 image using a 100-dim noise vector
        self.G = Sequential()
        
        # (100,) --> Layer 1 --> (7x7x256)
        self.G.add(Dense(7*7*256, input_dim = 100))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))
        self.G.add(Reshape((7,7,256)))
        
        # (7x7x256) --> Layer 2 --> (14x14x128)
        self.G.add(UpSampling2D(interpolation='nearest'))
        self.G.add(Conv2D(128, (5,5), padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))
        
        # (14x14x128) --> Layer 3 --> (14x14x64)
        self.G.add(Conv2D(64, (5,5), padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))
        
        # (14x14x64) --> Layer 4 --> (28x28x32)
        self.G.add(UpSampling2D(interpolation='nearest'))
        self.G.add(Conv2D(32, (5,5), padding='same'))
        self.G.add(BatchNormalization())
        self.G.add(Activation('relu'))
        
        # (28x28x32) --> Layer 5 --> (28x28x1)
        self.G.add(Conv2D(1, (5,5), padding='same'))
        self.G.add(Activation('sigmoid'))
        
        self.G.summary()
        return 1
    
    
    def create_discriminator_model(self):
        # Discriminator model
        # y_pred is 1 for real images, and 0 for fake images
        self.DM = Sequential()
        self.DM.add(self.D)
        optim = Adam(lr=adam_lr, decay=adam_decay)
        self.DM.compile(optimizer=optim,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return 1
        
        
    def create_combined_model(self):
        # Combined Adversarial model
        # y_pred is 1 always, as all generated images are fake
        for layer in self.D.layers:
            layer.trainable = False
        # self.D.trainable = False  # Discriminator shouldn't update weights 
        # when generator is trained.
        
        self.CM = Sequential()
        self.CM.add(self.G)
        
        self.CM.add(self.D)
        optim = Adam(lr=adam_lr, decay=adam_decay)
        self.CM.compile(optimizer=optim,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return 1
        
    
    def train(self, x_train, epochs, batch_size, k):
        # Train the GAN on [x_train]
        # k steps of optimizing D and 1 step of optimizing G
        # Each step has batch_size samples
        # Network weights, architecture get saved every 100th epoch
        # Randomly generated 25 samples saved every 100th epoch
        data_pointer = 0
        num_samples = x_train.shape[0]
        tic = time.time()
        y = np.ones((2*batch_size, 1))
        foldername = self.temp_loc
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        fid = open(foldername + "/metric_track.txt", "w")
        print("Epoch\tG_loss\tG_acc\tD_lossR\tD_lossF\tD_accR\tD_accF", file=fid)
        fid.close()
        self.generate_and_save("{}/gen_0.png".format(self.temp_loc))
        
        for epoch in range(1, epochs+1):
            print("Epoch {} Starts".format(epoch))
            while(True):
                for _ in range(k):
                    data_pointer_new = min(data_pointer + batch_size, 
                                           num_samples)
                    images_real = x_train[data_pointer: data_pointer_new, 
                                          :, :, :]
                    data_pointer = data_pointer_new
                    sample_size = images_real.shape[0]
                    noise = np.random.normal(0.0, 1.0, size=[sample_size, 100])
                    images_fake = self.G.predict(noise)  # generate fake images
                    D_loss_real = self.DM.train_on_batch(images_real,
                                                         np.ones((sample_size, 1)))
                    D_loss_fake = self.DM.train_on_batch(images_fake,
                                                         np.zeros((sample_size, 1)))
                    # print("D on Real: Loss:{}\tAcc:{}".format(D_loss_real[0], D_loss_real[1]))
                    # print("D on Fake: Loss:{}\tAcc:{}".format(D_loss_fake[0], D_loss_fake[1]))
                    if data_pointer >= num_samples:
                        # Whole data used
                        data_pointer = 0
                        break
                if data_pointer == 0:
                    break  # exit the while loop
                # 1 step of optimization for G
                noise = np.random.normal(0.0, 1.0, size=[batch_size*2, 100])
                G_loss = self.CM.train_on_batch(noise, y)
                # print("G: Loss:{}\tAcc:{}".format(G_loss[0], G_loss[1]))
            if epoch%10 == 0: # save 25 results generated by network at every
                # 10th epoch
                self.generate_and_save("{}/gen_{}.png".format(self.temp_loc, 
                                       epoch))
                self.save_net(self.temp_loc, epoch)  # save network data
                # self.save_to_pc(epoch)
                print("Results saved")
            #print("Epoch {} Ends\n----------------------------".format(epoch))
            toc = time.time()
            # Update data file. Format:
            # Epoch# G_loss G_acc D_real_loss D_fake_loss D_real_accuracy D_fake_accuracy
            images_real = x_train[np.random.randint(0,\
                                num_samples, size=batch_size), :, :, :]
            D_loss_real = self.DM.evaluate(images_real,\
                                           np.ones((batch_size, 1)))
            noise = np.random.normal(0.0, 1.0, size=[batch_size, 100])
            images_fake = self.G.predict(noise)  # generate fake images
            D_loss_fake = self.DM.evaluate(images_real,\
                                           np.zeros((batch_size, 1)))
            noise = np.random.normal(0.0, 1.0, size=[2*batch_size, 100])
            G_loss = self.CM.evaluate(noise, y)
            print("D on Real: Loss:{}\tAcc:{}".format(D_loss_real[0], D_loss_real[1]))
            print("D on Fake: Loss:{}\tAcc:{}".format(D_loss_fake[0], D_loss_fake[1]))
            print("G: Loss:{}\tAcc:{}".format(G_loss[0], G_loss[1]))
            print("Time To Complete: {}".format(get_formatted_time((toc-tic)/epoch * epochs)))
            self.save_metrics(epoch, G_loss, D_loss_real, D_loss_fake)
        
        self.save_net(self.temp_loc, epoch)
        # self.save_to_pc(epoch)
        print("Training completed")
        return 1
    
    
    def save_to_pc(self, epoch):
        print("{}/D{}.yaml".format(self.temp_loc, epoch))
        files.download("{}/D{}.yaml".format(self.temp_loc, epoch)) 
        files.download("{}/D{}.h5".format(self.temp_loc, epoch)) 
        files.download("{}/G{}.yaml".format(self.temp_loc, epoch)) 
        files.download("{}/G{}.h5".format(self.temp_loc, epoch)) 
        files.download("{}/gen_{}.png".format(self.temp_loc, epoch)) 
        files.download("{}/G{}.h5".format(self.temp_loc, epoch))
        files.download("{}/metric_track.txt".format(self.temp_loc))
        shutil.copy("{}/metric_track.txt".format(self.temp_loc), \
                    'drive/My Drive/Collaboratory/1_Basic_GAN/metric_track{}.txt'.format(epoch))
        return 1
    
    
    def save_metrics(self, epoch, G_loss, D_loss_real, D_loss_fake):
        # Save the metrics to file
        foldername = self.temp_loc
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        fid = open(foldername + "/metric_track.txt", "a")
        print("{:6d}\t{}\t{}\t{}\t{}\t{}\t{}".format(epoch, G_loss[0], \
              G_loss[1], D_loss_real[0], D_loss_fake[0], D_loss_real[1], \
              D_loss_fake[1]), file=fid)
        fid.close()
    
    
    def save_net(self, foldername, epoch):
        # Save the network and its weights  
        # serialize model to YAML
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        model_yaml = self.G.to_yaml()
        with open(foldername + "/G.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
            yaml_file.close()
        # serialize weights to HDF5
        self.G.save_weights(foldername + "/G.h5")
        
        model_yaml = self.D.to_yaml()
        with open(foldername + "/D.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
            yaml_file.close()
        # serialize weights to HDF5
        self.D.save_weights(foldername + "/D.h5")
        
        print("Saved models to disk")
        return 1
                    
                
    def generate_and_save(self, filename):
        # Generate 5x5 samples of generator
        if '/' in filename:
            foldername = filename[0:filename.rfind('/')]
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        noise = np.random.normal(0.0, 1.0, size=[25, 100])
        images_fake = self.G.predict(noise)  # generate fake images
        plt.figure(figsize=(5, 5))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            image = images_fake[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return 1


def get_formatted_time(seconds):
    # Returns the formatted time string in Hr:Mn:Sec
    hour = int(seconds / 3600)
    seconds -= hour * 3600
    minute = int(seconds / 60)
    seconds -= minute * 60
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds 