# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This library aims to create the building blocks required for Generative 
# Adversarial Network for image generation.
# References: (To be updated)

from keras.models import Sequential, Model
from keras.models import model_from_yaml
from keras.layers import Dense, Activation, Conv2D
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization, Flatten, Reshape
from keras.layers import UpSampling2D, Input, multiply
from keras.layers import Concatenate, RepeatVector, Lambda
from keras.backend import expand_dims, repeat_elements
from keras.utils import plot_model
import keras
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import time

adam_lr = 0.001
adam_decay = 0.0
embedding_shape = (10, 1024)
Ng = 128
Nd = 128
Md = 4
Nz = 100


def KL_loss(yTrue,yPred):
    mu = yPred[:, :, 0]
    sigma = yPred[:, :, 1]
    y = -sigma + 0.5 * (-1 + keras.backend.exp(2 * sigma) + keras.backend.square(mu))
    return keras.backend.mean(y)
    

class stack_GAN():
    def __init__(self, identity, gan1_image_width, gan2_image_width):
        self.GAN1 = GAN_type1('GAN1', gan1_image_width)
        self.GAN2 = GAN_type2('GAN2', gan2_image_width)
        self.create_models()  # create the training and testing models
        # List of models in stack_GAN:
        # self.stage1_Disc_model
        # self.stage1_Combined_model
        # self.stage1_Gen_model
        # self.stage2_Disc_model
        # self.stage2_Combined_model
        # self.end_to_end_Generate    
        self.temp_loc = identity
        self.round = 0  # the number of rounds training has happened
        
        
    def load_model(self, filename):
        # load YAML and create model
        yaml_file = open(filename + '.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(filename + ".h5")
        print("Loaded model from disk")
        return loaded_model
        
        
    def load_weights(self):
        self.stage1_Disc_model = self.load_model("stage1_Disc_model")
        self.stage1_Combined_model = self.load_model("stage1_Combined_model")
        self.stage1_Gen_model = self.load_model("stage1_Gen_model")
        self.stage2_Disc_model = self.load_model("stage2_Disc_model")
        self.stage2_Combined_model = self.load_model("stage2_Combined_model")
        self.end_to_end_Generate = self.load_model("end_to_end_Generate")
        return 1
        
        
    def create_models(self):
        # Inputs
        embedding_input = Input(shape=embedding_shape, name='text_emb')
        CA_noise_stage1 = Input(shape=(Ng,), name='CA_noise_stage1')
        gen_noise_stage1 = Input(shape=(Nz,), name='gen_noise_stage1')
        CA_noise_stage2 = Input(shape=(Ng,), name='CA_noise_stage2')
        lr_image_to_D_stage1 = Input(shape=(self.GAN1.width, self.GAN1.width, 3),\
                            name='lr_image_to_D_stage1')
        hr_image_to_D_stage2 = Input(shape=(self.GAN2.width, self.GAN2.width, 3),\
                            name='hr_image_to_D_stage2')
        
        # parts of the model
        optim = Adam(lr=adam_lr, decay=adam_decay)
        #-------------------
        # Stage 1 D
        D_stage1_out = self.GAN1.D([embedding_input, lr_image_to_D_stage1])
        self.stage1_Disc_model = Model(inputs=[embedding_input, \
                                               lr_image_to_D_stage1],\
                                        outputs=D_stage1_out)
        self.stage1_Disc_model.compile(optimizer=optim,\
                                       loss='binary_crossentropy',\
                                       metrics=['accuracy'])
        #----------------------
        # Stage 1 Combined
        [G_stage1_out, musigma1] = self.GAN1.G([embedding_input,\
                                    CA_noise_stage1,\
                                    gen_noise_stage1])
        stage1_D_mirror = Model(inputs=[embedding_input, \
                                        lr_image_to_D_stage1],\
                                        outputs=D_stage1_out)
        stage1_D_mirror.trainable = False
        combined_out_stage1 = stage1_D_mirror([embedding_input, G_stage1_out])
        self.stage1_Combined_model = Model(inputs = [embedding_input,\
                                    CA_noise_stage1,\
                                    gen_noise_stage1],\
                                    outputs = [combined_out_stage1, musigma1])
        self.stage1_Combined_model.compile(optimizer=optim,\
                                           loss=['binary_crossentropy', KL_loss],\
                                           metrics=['accuracy'])
        #-------------------------
        # Stage 1 G
        self.stage1_Gen_model = Model(inputs=[embedding_input,\
                                              CA_noise_stage1,\
                                              gen_noise_stage1],\
                                        outputs = G_stage1_out)
        self.stage1_Gen_model.trainable = False
        #-------------------------
        # Stage 2 D
        D_stage2_out = self.GAN2.D([embedding_input, hr_image_to_D_stage2])
        self.stage2_Disc_model = Model(inputs=[embedding_input, \
                                               hr_image_to_D_stage2],\
                                        outputs=D_stage2_out)
        self.stage2_Disc_model.compile(optimizer=optim,\
                                       loss='binary_crossentropy',\
                                       metrics=['accuracy'])
        #--------------------------
        # Stage 2 Combined
        stage2_Gen_lr_input = self.stage1_Gen_model([embedding_input,\
                                              CA_noise_stage1,\
                                              gen_noise_stage1])
        [G_stage2_out, musigma2] = self.GAN2.G([embedding_input,\
                                    CA_noise_stage2,\
                                    stage2_Gen_lr_input])
        stage2_D_mirror = Model(inputs=[embedding_input, \
                                        hr_image_to_D_stage2],\
                                        outputs=D_stage2_out)
        stage2_D_mirror.trainable = False
        combined_out_stage2 = stage2_D_mirror([embedding_input, G_stage2_out])
        self.stage2_Combined_model = Model(inputs = [embedding_input,\
                                                     CA_noise_stage1,\
                                                     gen_noise_stage1,\
                                                     CA_noise_stage2],\
                                            outputs = [combined_out_stage2, musigma2])
        self.stage2_Combined_model.compile(optimizer=optim,\
                                           loss=['binary_crossentropy', KL_loss],\
                                           metrics=['accuracy'])
        #---------------------------
        # End-to-end-generation
                
        self.end_to_end_Generate = Model(inputs=[embedding_input,\
                                                 CA_noise_stage1,\
                                                 gen_noise_stage1,\
                                                 CA_noise_stage2],
                                         outputs=G_stage2_out)
        self.end_to_end_Generate.trainable = False
        return 1
    
    
    def plot_models(self, ):
        plot_model(self.stage1_Disc_model, to_file='stage1_Disc_model.png')
        plot_model(self.stage1_Combined_model, to_file='stage1_Combined_model.png')
        plot_model(self.stage1_Gen_model, to_file='stage1_Gen_model.png')
        plot_model(self.stage2_Disc_model, to_file='stage2_Disc_model.png')
        plot_model(self.stage2_Combined_model, to_file='stage2_Combined_model.png')
        return 1
    
    
    def train(self, training_rounds, epochs_per_round, batch_size, k, \
              embedding_data, lr_imgs, hr_imgs):
        # Train the stacked GAN with text_embedding, lr_images and hr_images
        # k: number of iterations for optimizing D per iteration for optimizing
        # G
        # List of models in stack_GAN:
        # self.stage1_Disc_model
        # self.stage1_Combined_model
        # self.stage1_Gen_model
        # self.stage2_Disc_model
        # self.stage2_Combined_model
        # self.end_to_end_Generate
        
        self.lr_images = lr_imgs
        self.hr_images = hr_imgs
        self.text_embedding = embedding_data
        
        self.round = 0
        self.epochs_per_round = epochs_per_round
        print("Training started")
        for _ in range(int(training_rounds/2)):
            self.train_GAN1(epochs_per_round, batch_size, k)
            self.round += 1
            self.train_GAN2(epochs_per_round, batch_size, k)
            self.round += 1
        return 1
        
    
    def train_GAN1(self, epochs, batch_size, k):
        # Randomly generated 25 samples saved every 10th epoch
        data_pointer = 0
        num_samples = self.text_embedding.shape[0]
        tic = time.time()
        foldername = self.temp_loc
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        fid = open(foldername + "/metric_track.txt", "w")
        print("Epoch\tD1_loss_R\tD1_acc_R\tD1_loss_F2\tD1_acc_F2\tD1_loss_F3"+\
              "\tD1_acc_F3\tG1_loss\tG1_LossBCE\tG1_LossKL\tG1_accBCE\tG1_accKL"+\
              "\tD2_loss_R\tD2_acc_R\tD2_loss_F2\tD2_acc_F2\tD2_loss_F3"+\
              "\tD2_acc_F3\tG2_loss\tG2_LossBCE\tG2_LossKL\tG2_accBCE\tG2_accKL", \
              file=fid)
        fid.close()
        self.generate_and_save("{}/GAN#_gen_{}_0".format(self.temp_loc,\
                                                       self.round), 1)
        descr_shape_init = (0, embedding_shape[0], embedding_shape[1])
        
        for epoch in range(1, epochs+1):
            print("Round {}\tGAN1\tEpoch {}".format(self.round, epoch))
            while(True):
                descr_true_accumulated = np.array([])  # used for generator
                descr_true_accumulated = descr_true_accumulated.reshape(descr_shape_init)
                sample_size_for_G = 0
                for _ in range(k):
                    # k steps of training for D
                    # try to use k=1 always
                    data_pointer_new = min(data_pointer + batch_size, 
                                           num_samples)
                    images_real = self.lr_images[data_pointer: data_pointer_new, 
                                          :, :, :]
                    images_real = self.condition_images(images_real)
                    sample_size = images_real.shape[0]
                    sample_size_for_G += sample_size
                    descr_true = self.text_embedding[data_pointer: data_pointer_new, 
                                          :, :]
                    descr_true_accumulated = np.concatenate((descr_true_accumulated,\
                                                             descr_true), axis = 0)
                    
                    data_pointer = data_pointer_new
                    # CA noise
                    ca_noise = np.zeros((sample_size, Ng))
                    # Generator noise
                    gen_noise = np.random.normal(0.0, 1.0, size=[sample_size, Nz])
                    # generate fake images
                    images_fake = self.stage1_Gen_model.predict([descr_true,\
                                                                 ca_noise,\
                                                                 gen_noise])
                    # Randomly choose some descriptions from the data set. We
                    # hope that as they are random, a large number of true 
                    # descriptions won't come up
                    descr_fake = self.text_embedding[np.random.randint(0,\
                                        num_samples, size=sample_size), :, :]
                    # D training consists of 3 parts
                    # 1. real image true description
                    # 2. fake image true description
                    # 3. fake image mismatched description
                    # 1. real image true description:
                    D1_loss_real = self.stage1_Disc_model.train_on_batch([descr_true,\
                                                                         images_real],\
                                                         np.ones((sample_size, 1)))
                    
                    # 2. fake image true description: 
                    D1_loss_fake2 = self.stage1_Disc_model.train_on_batch([descr_true,\
                                                                          images_fake],\
                                                         np.zeros((sample_size, 1)))
                    
                    # 3. fake image mismatched description
                    D1_loss_fake3 = self.stage1_Disc_model.train_on_batch([descr_fake,\
                                                                          images_fake],\
                                                         np.zeros((sample_size, 1)))
                    # print("D on Real: Loss:{}\tAcc:{}".format(D_loss_real[0], D_loss_real[1]))
                    # print("D on Fake: Loss:{}\tAcc:{}".format(D_loss_fake[0], D_loss_fake[1]))
                    if data_pointer >= num_samples:
                        # Whole data used
                        data_pointer = 0
                        break
                # 1 step of optimization for G
                # CA noise
                ca_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Ng])
                # Generator noise
                gen_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Nz])
                
                G1_loss = self.stage1_Combined_model.train_on_batch([descr_true_accumulated,\
                                                ca_noise, gen_noise],\
                                                [np.ones((sample_size_for_G, 1)), \
                                                 np.zeros((sample_size_for_G, Ng, 2))])
                # CA noise
                ca_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Ng])
                # Generator noise
                gen_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Nz])
                
                G1_loss = self.stage1_Combined_model.train_on_batch([descr_true_accumulated,\
                                                ca_noise, gen_noise],\
                                                [np.ones((sample_size_for_G, 1)), \
                                                 np.zeros((sample_size_for_G, Ng, 2))])
                # CA noise
                ca_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Ng])
                # Generator noise
                gen_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Nz])
                
                G1_loss = self.stage1_Combined_model.train_on_batch([descr_true_accumulated,\
                                                ca_noise, gen_noise],\
                                                [np.ones((sample_size_for_G, 1)), \
                                                 np.zeros((sample_size_for_G, Ng, 2))])

                # print("G: Loss:{}\tAcc:{}".format(G_loss[0], G_loss[1]))
                
                if data_pointer == 0:
                    break  # exit the while loop
                
            if epoch%10 == 0: # save 25 results generated by network at every
                # 10th epoch
                self.generate_and_save("{}/GAN#_gen_{}_{}.png".format(self.temp_loc,\
                                       self.round, epoch), 1)
                self.save_net(self.temp_loc, epoch)  # save network data
                print("Results saved")
            toc = time.time()
            # Update data file. Format:
            # Epoch# G_loss G_acc D_real_loss D_fake_loss D_real_accuracy D_fake_accuracy
            assessment = self.assess_network(batch_size)
            self.save_metrics(assessment, epoch)
        
        self.save_net(self.temp_loc, epoch)
        # self.save_to_pc(epoch)
        print("GAN1 Training Round{} completed".format(self.round))
        return 1
    
    
    def train_GAN2(self, epochs, batch_size, k):
        # Randomly generated 25 samples saved every 10th epoch
        data_pointer = 0
        num_samples = self.text_embedding.shape[0]
        tic = time.time()
        foldername = self.temp_loc
        if not os.path.exists(foldername):
            os.makedirs(foldername)
#        fid = open(foldername + "/metric_track_GAN2.txt", "w")
#        print("Epoch\tG_loss\tG_acc\tD_lossR\tD_lossF\tD_accR\tD_accF", file=fid)
#        fid.close()
        self.generate_and_save("{}/GAN#_gen_{}_0".format(self.temp_loc,\
                                                       self.round), 2)
        descr_shape_init = (0, embedding_shape[0], embedding_shape[1])
        
        for epoch in range(1, epochs+1):
            print("Round {}\tGAN2\tEpoch {}".format(self.round, epoch))
            while(True):
                descr_true_accumulated = np.array([])  # used for generator
                descr_true_accumulated = descr_true_accumulated.reshape(descr_shape_init)
                sample_size_for_G = 0
                for _ in range(k):
                    # k steps of training for D
                    # try to use k=1 always
                    data_pointer_new = min(data_pointer + batch_size, 
                                           num_samples)
                    images_real = self.hr_images[data_pointer: data_pointer_new, 
                                          :, :, :]
                    images_real = self.condition_images(images_real)
                    sample_size = images_real.shape[0]
                    sample_size_for_G += sample_size
                    descr_true = self.text_embedding[data_pointer: data_pointer_new, 
                                          :, :]
                    descr_true_accumulated = np.concatenate((descr_true_accumulated,\
                                                             descr_true), axis = 0)
                    
                    data_pointer = data_pointer_new
                    # CA noise
                    ca_noise1 = np.zeros((sample_size, Ng))
                    ca_noise2 = np.zeros((sample_size, Ng))
                    # Generator noise
                    gen_noise = np.random.normal(0.0, 1.0, size=[sample_size, Nz])
                    # generate fake images
                    images_fake = self.end_to_end_Generate.predict([descr_true,\
                                                                 ca_noise1,\
                                                                 gen_noise,\
                                                                 ca_noise2])
                    # Randomly choose some descriptions from the data set. We
                    # hope that as they are random, a large number of true 
                    # descriptions won't come up
                    descr_fake = self.text_embedding[np.random.randint(0,\
                                        num_samples, size=sample_size), :, :]
                    # D training consists of 3 parts
                    # 1. real image true description
                    # 2. fake image true description
                    # 3. fake image mismatched description
                    # 1. real image true description:
                    D2_loss_real = self.stage2_Disc_model.train_on_batch([descr_true,\
                                                                         images_real],\
                                                         np.ones((sample_size, 1)))
                    
                    # 2. fake image true description: 
                    D2_loss_fake2 = self.stage2_Disc_model.train_on_batch([descr_true,\
                                                                          images_fake],\
                                                         np.zeros((sample_size, 1)))
                    
                    # 3. fake image mismatched description
                    D2_loss_fake3 = self.stage2_Disc_model.train_on_batch([descr_fake,\
                                                                          images_fake],\
                                                         np.zeros((sample_size, 1)))
                    # print("D on Real: Loss:{}\tAcc:{}".format(D_loss_real[0], D_loss_real[1]))
                    # print("D on Fake: Loss:{}\tAcc:{}".format(D_loss_fake[0], D_loss_fake[1]))
                    if data_pointer >= num_samples:
                        # Whole data used
                        data_pointer = 0
                        break
                # 1 step of optimization for G
                # CA noise
                ca_noise1 = np.zeros((sample_size_for_G, Ng))
                ca_noise2 = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Ng])
                # Generator noise
                gen_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Nz])
                
                G2_loss = self.stage2_Combined_model.train_on_batch([descr_true_accumulated,\
                                                ca_noise1, gen_noise, ca_noise2],\
                                                [np.ones((sample_size_for_G, 1)), \
                                                 np.zeros((sample_size_for_G, Ng, 2))])
                G2_loss = self.stage2_Combined_model.train_on_batch([descr_true_accumulated,\
                                                ca_noise1, gen_noise, ca_noise2],\
                                                [np.ones((sample_size_for_G, 1)), \
                                                 np.zeros((sample_size_for_G, Ng, 2))])
                G2_loss = self.stage2_Combined_model.train_on_batch([descr_true_accumulated,\
                                                ca_noise1, gen_noise, ca_noise2],\
                                                [np.ones((sample_size_for_G, 1)), \
                                                 np.zeros((sample_size_for_G, Ng, 2))])
                # print("G: Loss:{}\tAcc:{}".format(G_loss[0], G_loss[1]))
                
                if data_pointer == 0:
                    break  # exit the while loop
                
            if epoch%10 == 0: # save 25 results generated by network at every
                # 10th epoch
                self.generate_and_save("{}/GAN#_gen_{}_{}.png".format(self.temp_loc,\
                                       self.round, epoch), 2)
                self.save_net(self.temp_loc, epoch)  # save network data
                print("Results saved")
            toc = time.time()
            # Update data file. Format:
            # Epoch# G_loss G_acc D_real_loss D_fake_loss D_real_accuracy D_fake_accuracy
            assessment = self.assess_network(batch_size)
            self.save_metrics(assessment, epoch)
        
        self.save_net(self.temp_loc, epoch)
        # self.save_to_pc(epoch)
        print("GAN2 Training Round{} completed".format(self.round))
        return 1
    
    
    def condition_images(self, images):
        # condition images for usage
        return (images/255.0 - 0.5)*2
    
    
    def save_metrics(self, assessment, epoch):
        # Saves the matrics and losses to file
        foldername = self.temp_loc
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        fid = open(foldername + "/metric_track.txt", "a")
        epoch = self.epochs_per_round * self.round + epoch
        text = "{:10d}".format(epoch)
        for data in assessment:
            for metric in data:
                text = "{}\t{:6.4f}".format(text, metric)
        print(text, file=fid)
        fid.close()
    
    
    def assess_network(self, sample_size):
        # Assess the stack-GAN for the metrics and losses and return them
        num_samples = self.text_embedding.shape[0]
        random_indices = np.random.randint(0, num_samples, size=sample_size)
        descr_true = self.text_embedding[random_indices, :, :]
        #GAN1----------------------------------------------------
        images_true_lr = self.lr_images[random_indices, :, :, :]
        images_true_lr = self.condition_images(images_true_lr)
        images_true_hr = self.hr_images[random_indices, :, :, :]
        images_true_hr = self.condition_images(images_true_hr)
        # CA noise
        ca_noise = np.zeros((sample_size, Ng))
        # Generator noise
        gen_noise = np.random.normal(0.0, 1.0, size=[sample_size, Nz])
        # generate fake images
        images_fake = self.stage1_Gen_model.predict([descr_true,\
                                                     ca_noise,\
                                                     gen_noise])
        # Randomly choose some descriptions from the data set. We
        # hope that as they are random, a large number of true 
        # descriptions won't come up
        descr_fake = self.text_embedding[np.random.randint(0,\
                                num_samples, size=sample_size), :, :]
        D1_loss_real = self.stage1_Disc_model.evaluate([descr_true,\
                                                      images_true_lr],\
                                                    np.ones((sample_size, 1)))
                    
        # 2. fake image true description: 
        D1_loss_fake2 = self.stage1_Disc_model.evaluate([descr_true,\
                                                       images_fake],\
                                                    np.zeros((sample_size, 1)))
                    
        # 3. fake image mismatched description
        D1_loss_fake3 = self.stage1_Disc_model.evaluate([descr_fake,\
                                                       images_fake],\
                                                    np.zeros((sample_size, 1)))
        # CA noise
        ca_noise = np.random.normal(0.0, 1.0, size=[sample_size, Ng])
        # Generator noise
        gen_noise = np.random.normal(0.0, 1.0, size=[sample_size, Nz])
        
        G1_loss = self.stage1_Combined_model.evaluate([descr_true,\
                                            ca_noise, gen_noise],\
                                            [np.ones((sample_size, 1)), \
                                             np.zeros((sample_size, Ng, 2))])
        #GAN2---------------------------------------------------------------
        # CA noise
        ca_noise1 = np.zeros((sample_size, Ng))
        ca_noise2 = np.zeros((sample_size, Ng))
        # Generator noise
        # gen_noise = np.random.normal(0.0, 1.0, size=[sample_size, Nz])
        # generate fake images
        images_fake = self.end_to_end_Generate.predict([descr_true,\
                                                        ca_noise1,\
                                                        gen_noise,\
                                                        ca_noise2])
        # Randomly choose some descriptions from the data set. We
        # hope that as they are random, a large number of true 
        # descriptions won't come up
        # descr_fake = self.text_embedding[np.random.randint(0,\
        #                            num_samples, size=sample_size), :, :]
        # D training consists of 3 parts
        # 1. real image true description
        # 2. fake image true description
        # 3. fake image mismatched description
        # 1. real image true description:
        D2_loss_real = self.stage2_Disc_model.evaluate([descr_true,\
                                                        images_true_hr],\
                                                    np.ones((sample_size, 1)))
                    
        # 2. fake image true description: 
        D2_loss_fake2 = self.stage2_Disc_model.evaluate([descr_true,\
                                                         images_fake],\
                                                    np.zeros((sample_size, 1)))
                    
        # 3. fake image mismatched description
        D2_loss_fake3 = self.stage2_Disc_model.evaluate([descr_fake,\
                                                         images_fake],\
                                                    np.zeros((sample_size, 1)))
        # CA noise
        # ca_noise1 = np.zeros((sample_size, Ng))
        ca_noise2 = np.random.normal(0.0, 1.0, size=[sample_size, Ng])
        # Generator noise
        # gen_noise = np.random.normal(0.0, 1.0, size=[sample_size_for_G, Nz])
        
        G2_loss = self.stage2_Combined_model.evaluate([descr_true,\
                                            ca_noise1, gen_noise, ca_noise2],\
                                            [np.ones((sample_size, 1)), \
                                             np.zeros((sample_size, Ng, 2))])
        print("D1 on Real: Loss:{}\tAcc:{}".format(D1_loss_real[0], D1_loss_real[1]))
        print("D1 on Fake2: Loss:{}\tAcc:{}".format(D1_loss_fake2[0], D1_loss_fake2[1]))
        print("D1 on Fake3: Loss:{}\tAcc:{}".format(D1_loss_fake3[0], D1_loss_fake3[1]))
        print("G1: Loss:{}\tLoss1:{}\tLoss2:{}\tAcc1:{}\tAcc2:{}".format(G1_loss[0], \
              G1_loss[1], G1_loss[2], G1_loss[3], G1_loss[4]))
        
        print("D2 on Real: Loss:{}\tAcc:{}".format(D2_loss_real[0], D2_loss_real[1]))
        print("D2 on Fake2: Loss:{}\tAcc:{}".format(D2_loss_fake2[0], D2_loss_fake2[1]))
        print("D2 on Fake3: Loss:{}\tAcc:{}".format(D2_loss_fake3[0], D2_loss_fake3[1]))
        print("G2: Loss:{}\tLoss1:{}\tLoss2:{}\tAcc1:{}\tAcc2:{}".format(G2_loss[0], \
              G2_loss[1], G2_loss[2], G2_loss[3], G2_loss[4]))
        
        # print("len_G1_loss:{}\tlen_G2_loss:{}".format(len(G1_loss), len(G2_loss)))
        # print("Time To Complete: {}".format(get_formatted_time((toc-tic)/epoch * epochs)))
        # self.save_metrics(epoch, G_loss, D_loss_real, D_loss_fake)
        return [D1_loss_real, D1_loss_fake2, D1_loss_fake3, G1_loss,\
                D2_loss_real, D2_loss_fake2, D2_loss_fake3, G2_loss]
    
    
    def save_net(self, foldername, epoch):
        # Save the network and its weights  
        # serialize following models to YAML
        #------------------------------------
        # self.stage1_Disc_model
        # self.stage1_Combined_model
        # self.stage1_Gen_model
        # self.stage2_Disc_model
        # self.stage2_Combined_model
        # self.end_to_end_Generate
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        
        self.save_model(self.stage1_Disc_model, "/stage1_Disc_model")
        self.save_model(self.stage1_Combined_model, "/stage1_Combined_model")
        self.save_model(self.stage1_Gen_model, "/stage1_Gen_model")
        self.save_model(self.stage2_Disc_model, "/stage2_Disc_model")
        self.save_model(self.stage2_Combined_model, "/stage2_Combined_model")
        self.save_model(self.end_to_end_Generate, "/end_to_end_Generate")
        
        print("Saved models to disk")
        return 1
    
    
    def save_model(self, model, name):
        model_yaml = model.to_yaml()
        foldername = self.temp_loc
        with open(foldername + name + ".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
            yaml_file.close()
        # serialize weights to HDF5
        model.save_weights(foldername + name + ".h5")
    
    
    def generate_and_save(self, filename, gan_select):
        # Generate 5x5 samples of generator specified by gan_select
        # gan_select not used right now. May be later.
        no_of_samples_saved = 25
        filename1 = filename.replace('#', str(1))
        filename2 = filename.replace('#', str(2))
        filename3 = filename.replace('GAN#', 'REAL')
        if '/' in filename:
            foldername = filename[0:filename.rfind('/')]
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        # Random text embeddings and their corresponding real images
        num_samples = self.text_embedding.shape[0]
        random_indices = np.random.randint(0, num_samples, size=no_of_samples_saved)
        descr_true = self.text_embedding[random_indices, :, :]
        # Save the true images too, so as to compare.
        images_true = self.hr_images[random_indices, :, :, :]
        images_true = self.condition_images(images_true)
        # CA noise
        ca_noise = np.zeros((no_of_samples_saved, Ng))
        # Generator noise
        gen_noise = np.random.normal(0.0, 1.0, size=[no_of_samples_saved, Nz])
        # generate fake images
        images_stage1 = self.stage1_Gen_model.predict([descr_true,\
                                        ca_noise, gen_noise])
        images_stage2 = self.end_to_end_Generate.predict([descr_true,\
                                                 ca_noise,\
                                                 gen_noise,\
                                                 ca_noise])
        plt.figure(figsize=(5, 5))
        for i in range(no_of_samples_saved):
            plt.subplot(5, 5, i+1)
            image = images_true[i, :, :, :]
            image = np.reshape(image, [256, 256, 3])
            image = (image+1)/2
            plt.imshow(image)
            plt.axis('off')
        #plt.tight_layout()
        plt.savefig(filename3)
        plt.close('all')
        
        plt.figure(figsize=(5, 5))
        for i in range(no_of_samples_saved):
            plt.subplot(5, 5, i+1)
            image = images_stage1[i, :, :, :]
            image = np.reshape(image, [64, 64, 3])
            image = (image+1)/2
            plt.imshow(image)
            plt.axis('off')
        #plt.tight_layout()
        plt.savefig(filename1)
        plt.close('all')
        
        plt.figure(figsize=(5, 5))
        for i in range(no_of_samples_saved):
            plt.subplot(5, 5, i+1)
            image = images_stage2[i, :, :, :]
            image = np.reshape(image, [256, 256, 3])
            image = (image+1)/2
            plt.imshow(image)
            plt.axis('off')
        #plt.tight_layout()
        plt.savefig(filename2)
        plt.close('all')
        
        return 1


class GAN_type1():
    # GAN of type 1 that generates color image of shape 
    # image_widthximage_width
    def __init__(self, identity, image_width):
        # Make sure to create a folder with name same as identity
        # so that data related to the network can be stored there
        self.width = image_width  # 64
        self.D = None
        self.G = None
        self.DM = None
        self.CM = None
        self.create_discriminator()  # Create discriminator
        self.create_generator()  # Create generator
#        self.create_discriminator_model()  # Create the discriminator model
#        self.create_combined_model()  # Create the combined model
        self.temp_loc = identity  # an identifier for the GAN. Temporary 
        # data are saved to the folder named from the identifier
        
    
    def create_discriminator(self):        
        # Discriminator takes in 64x64x3 Real/Generated image data and tells
        # if it is fake or real (fake = 0, real = 1)
        
        text_emb = Input(shape=embedding_shape, name='text_embedding_D')
        image_input = Input(shape=(self.width, self.width, 3),\
                            name='image_D')
        
        downsampler = Sequential()
        
        # (64x64x3) --> Layer 1 --> (32x32x64)
        downsampler.add(Conv2D(64, (4, 4), strides=2, padding='same', 
                          input_shape=(self.width, self.width, 3)))
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (32x32x64) --> Layer 2 --> (16x16x128)
        downsampler.add(Conv2D(128, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (16x16x128) --> Layer 3 --> (8x8x256)
        downsampler.add(Conv2D(256, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (8x8x256) --> Layer 4 --> (4x4x512)
        downsampler.add(Conv2D(512, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        
        y1 = downsampler(image_input)
        flattened_emb = Flatten()(text_emb)
        compressed_emb = Dense(Nd)(flattened_emb)
        
        replica = RepeatVector(Md)(compressed_emb)
        replica = Lambda(lambda x: expand_dims(x, axis=1))(replica)
        replica = Lambda(lambda x: repeat_elements(x, Md, axis=1))(replica)
        
        joined = Concatenate()([y1, replica])
                
        joined = Conv2D(8, (1, 1))(joined)
        last = Flatten()(joined)
        last = Dense(1)(last)
        out = Activation('sigmoid')(last)
        # print(type(out), text_emb.shape, type(image_input))
        
        self.D = Model(inputs=[text_emb, image_input], \
                      outputs=out)
        
        downsampler.summary()        
        self.D.summary()
        
        return 1
    
    
    def create_generator(self):
        # Generator generates a 64x64x3 image using a Nz-dim noise vector
        
        # Embedded text input
        text_emb = Input(shape=embedding_shape, name='text_embedding_G')
        ca_noise = Input(shape=(Ng,), name='CA_noise')  # CA for conditional aug.
        gen_noise = Input(shape=(Nz,), name='gen_noise')
        flat_emb = Flatten()(text_emb)
        # Conditional augmentation
        mu0 = Dense(Ng)(flat_emb)
        sigma0 = Dense(Ng)(flat_emb)
        noisy_sigma = multiply([sigma0, ca_noise])
        c_cap = keras.layers.add([mu0, noisy_sigma])
        
        y_in = Concatenate()([c_cap, gen_noise])
        
        mu0 = Reshape((Ng, 1))(mu0)
        sigma0 = Reshape((Ng, 1))(sigma0)
        musigma = Concatenate()([mu0, sigma0])

        upsampler = Sequential()
        s0 = int(self.width/16)
        # (Ng+Nz,) --> Layer 1 --> (s0,s0,1024)
        upsampler.add(Dense(s0*s0*1024, input_dim = Ng+Nz))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        upsampler.add(Reshape((s0,s0,1024)))
        
        # (s0,s0,1024) --> Layer 2 --> (s1,s1,512)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(512, (3,3), padding='same'))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        
        # (s1,s1,512) --> Layer 3 --> (s2,s2,256)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(256, (3,3), padding='same'))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        
        # (s2,s2,256) --> Layer 4 --> (s3,s3,128)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(128, (3,3), padding='same'))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        
        # (s3,s3,128) --> Layer 5 --> (s4,s4,3)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(3, (3,3), padding='same'))
        upsampler.add(Activation('tanh'))
        
        y_out = upsampler(y_in)
        
        self.G = Model(inputs=[text_emb, ca_noise, gen_noise], \
                      outputs=[y_out, musigma])
        
        upsampler.summary()
        self.G.summary()
        return 1
        
#    def create_discriminator_model(self):
#        # Discriminator model
#        # y_pred is 1 for real images, and 0 for fake images
#        self.DM = Sequential()
#        self.DM.add(self.D)
#        optim = Adam(lr=adam_lr, decay=adam_decay)
#        self.DM.compile(optimizer=optim,
#                        loss='binary_crossentropy',
#                        metrics=['accuracy'])
#        return 1
        
        
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
    
    
#    def save_to_pc(self, epoch):
#        print("{}/D{}.yaml".format(self.temp_loc, epoch))
#        files.download("{}/D{}.yaml".format(self.temp_loc, epoch)) 
#        files.download("{}/D{}.h5".format(self.temp_loc, epoch)) 
#        files.download("{}/G{}.yaml".format(self.temp_loc, epoch)) 
#        files.download("{}/G{}.h5".format(self.temp_loc, epoch)) 
#        files.download("{}/gen_{}.png".format(self.temp_loc, epoch)) 
#        files.download("{}/G{}.h5".format(self.temp_loc, epoch))
#        files.download("{}/metric_track.txt".format(self.temp_loc))
#        shutil.copy("{}/metric_track.txt".format(self.temp_loc), \
#                    'drive/My Drive/Collaboratory/1_Basic_GAN/metric_track{}.txt'.format(epoch))
#        return 1
    
    
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
    

class GAN_type2():
    def __init__(self, identity, image_width):
        # Make sure to create a folder with name same as identity
        # so that data related to the network can be stored there
        self.width = image_width  # 256
        self.D = None
        self.G = None
        self.DM = None
        self.CM = None
        self.create_discriminator()  # Create discriminator
        self.create_generator()  # Create generator
#        self.create_discriminator_model()  # Create the discriminator model
#        self.create_combined_model()  # Create the combined model
        self.temp_loc = identity  # an identifier for the GAN. Temporary 
        # data are saved to the folder named from the identifier
        
    
    def create_discriminator(self):        
        # Discriminator takes in 256x256x3 Real/Generated image data and tells
        # if it is fake or real (fake = 0, real = 1)
        
        text_emb = Input(shape=embedding_shape, name='text_embedding_D')
        image_input = Input(shape=(self.width, self.width, 3),\
                            name='image_D')
        
        downsampler = Sequential()
        
        # (256x256x3) --> Layer 1 --> (128x128x16)
        downsampler.add(Conv2D(16, (4, 4), strides=2, padding='same',
                               input_shape=(self.width, self.width, 3)))
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (128x128x16) --> Layer 2 --> (64x64x32)
        downsampler.add(Conv2D(32, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (64x64x32) --> Layer 3 --> (32x32x64)
        downsampler.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (32x32x64) --> Layer 4 --> (16x16x128)
        downsampler.add(Conv2D(128, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (16x16x128) --> Layer 5 --> (8x8x256)
        downsampler.add(Conv2D(256, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (8x8x256) --> Layer 6 --> (4x4x512)
        downsampler.add(Conv2D(512, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        
        y1 = downsampler(image_input)
        flattened_emb = Flatten()(text_emb)
        compressed_emb = Dense(Nd)(flattened_emb)
        
        replica = RepeatVector(Md)(compressed_emb)
        replica = Lambda(lambda x: expand_dims(x, axis=1))(replica)
        replica = Lambda(lambda x: repeat_elements(x, Md, axis=1))(replica)
        
        joined = Concatenate()([y1, replica])
                
        joined = Conv2D(8, (1, 1))(joined)
        last = Flatten()(joined)
        last = Dense(1)(last)
        out = Activation('sigmoid')(last)
        # print(type(out), text_emb.shape, type(image_input))
        
        self.D = Model(inputs=[text_emb, image_input], \
                      outputs=out)
        
        downsampler.summary()        
        self.D.summary()
        
        return 1
    
    
    def create_generator(self):
        # Generator generates a 64x64x3 image using a Nz-dim noise vector
        in_width = int(self.width/4)
        # Embedded text input
        text_emb = Input(shape=embedding_shape, name='text_embedding_G')
        ca_noise = Input(shape=(Ng,), name='CA_noise')  # CA for conditional aug.
        lr_input = Input(shape=(in_width, in_width, 3), name='lr_input_G')
        flat_emb = Flatten()(text_emb)
        # Conditional augmentation
        mu0 = Dense(Ng)(flat_emb)
        sigma0 = Dense(Ng)(flat_emb)
        noisy_sigma = multiply([sigma0, ca_noise])
        c_cap = keras.layers.add([mu0, noisy_sigma])
        
        mu0 = Reshape((Ng, 1))(mu0)
        sigma0 = Reshape((Ng, 1))(sigma0)
        musigma = Concatenate()([mu0, sigma0])
        
        replica = RepeatVector(16)(c_cap)
        replica = Lambda(lambda x: expand_dims(x, axis=1))(replica)
        replica = Lambda(lambda x: repeat_elements(x, 16, axis=1))(replica)
        
        downsampler = Sequential()
        
        # (64x64x3) --> Layer 1 --> (64x64x128)
        downsampler.add(Conv2D(128, (4, 4), strides=1, padding='same',
                               input_shape=(in_width, in_width, 3)))
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (64x64x128) --> Layer 2 --> (32x32x256)
        downsampler.add(Conv2D(256, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        # (32x32x256) --> Layer 3 --> (16x16x512)
        downsampler.add(Conv2D(512, (4, 4), strides=2, padding='same'))
        downsampler.add(BatchNormalization())
        downsampler.add(LeakyReLU(alpha=0.2))
        
        downsampled_image = downsampler(lr_input)
        
        joined = Concatenate()([downsampled_image, replica])
        
        y = residual_unit(joined, 640)
        y = residual_unit(joined, 640)        

        upsampler = Sequential()
        s0 = int(self.width/16)
        # print("Confirm if this is 16: {}".format(s0))
        
        # (s0,s0,640) --> Layer 1 --> (s1,s1,128)
        upsampler.add(UpSampling2D(interpolation='nearest',\
                                   input_shape=(s0, s0, 640)))
        upsampler.add(Conv2D(128, (3,3), padding='same'))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        
        # (s1,s1,128) --> Layer 3 --> (s2,s2,64)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(64, (3,3), padding='same'))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        
        # (s2,s2,64) --> Layer 4 --> (s3,s3,32)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(32, (3,3), padding='same'))
        upsampler.add(BatchNormalization())
        upsampler.add(Activation('relu'))
        
        # (s3,s3,32) --> Layer 5 --> (s4,s4,3)
        upsampler.add(UpSampling2D(interpolation='nearest'))
        upsampler.add(Conv2D(3, (3,3), padding='same'))
        upsampler.add(Activation('tanh'))
        
        y_out = upsampler(y)
        
        self.G = Model(inputs=[text_emb, ca_noise, lr_input], \
                      outputs=[y_out, musigma])
        
        upsampler.summary()
        self.G.summary()
        return 1
    
    
#    def create_discriminator_model(self):
#        # Discriminator model
#        # y_pred is 1 for real images, and 0 for fake images
#        self.DM = Sequential()
#        self.DM.add(self.D)
#        optim = Adam(lr=adam_lr, decay=adam_decay)
#        self.DM.compile(optimizer=optim,
#                        loss='binary_crossentropy',
#                        metrics=['accuracy'])
#        return 1
        
        
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
    
    
#    def save_to_pc(self, epoch):
#        print("{}/D{}.yaml".format(self.temp_loc, epoch))
#        files.download("{}/D{}.yaml".format(self.temp_loc, epoch)) 
#        files.download("{}/D{}.h5".format(self.temp_loc, epoch)) 
#        files.download("{}/G{}.yaml".format(self.temp_loc, epoch)) 
#        files.download("{}/G{}.h5".format(self.temp_loc, epoch)) 
#        files.download("{}/gen_{}.png".format(self.temp_loc, epoch)) 
#        files.download("{}/G{}.h5".format(self.temp_loc, epoch))
#        files.download("{}/metric_track.txt".format(self.temp_loc))
#        shutil.copy("{}/metric_track.txt".format(self.temp_loc), \
#                    'drive/My Drive/Collaboratory/1_Basic_GAN/metric_track{}.txt'.format(epoch))
#        return 1
    
    
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


def residual_unit(in_tensor, channels):
    # Residual unit with "same" convolution
    y = Conv2D(channels, (3,3), padding='same')(in_tensor)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(channels, (3,3), padding='same')(y)
    y = BatchNormalization()(y)
    y = keras.layers.add([in_tensor, y])
    y = Activation('relu')(y)
    return y
    

def get_formatted_time(seconds):
    # Returns the formatted time string in Hr:Mn:Sec
    hour = int(seconds / 3600)
    seconds -= hour * 3600
    minute = int(seconds / 60)
    seconds -= minute * 60
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds 