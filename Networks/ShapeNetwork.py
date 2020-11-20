from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D, Concatenate, UpSampling2D, Input, Conv2DTranspose, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import os
import numpy as np
import shutil


class ShapeNetwork:
    def __init__(self, config):

        shape = (config['dataset']['image_height'],
                 config['dataset']['image_width'], 6)
        regularization_weight = config['model']['regularization']
        shape_size = config['model']['shape_size']
        self.shape_size = config['model']['shape_size']
        alpha = 0.3

        self.regularization = l2(regularization_weight)

        layer_dim = int(shape[0]*shape[1]/(16*16))
        z_dim = (int(shape[0]//16), int(shape[1]//16))

        layer_depth = int(np.ceil(shape_size / layer_dim))

        input_img = Input(shape, name="image_input")

        x1 = Conv2D(8, (5, 5),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(input_img)
        x1 = Conv2D(8, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x1)

        x2 = Conv2D(16, (5, 5), strides=(2, 2),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x1)

        x2 = Conv2D(16, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x2)
        x2 = Conv2D(16, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x2)

        x3 = Conv2D(32, (5, 5), strides=(2, 2),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x2)

        x3 = Conv2D(32, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x3)

        x3 = Conv2D(32, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x3)

        x4 = Conv2D(32, (5, 5), strides=(2, 2),  activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x3)

        x4 = Conv2D(32, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x4)

        x4 = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x4)

        x5 = Conv2D(64, (5, 5), strides=(2, 2),   activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x4)

        x5 = Conv2D(64, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                    kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x5)

        x5_cnn = Conv2D(64, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                        kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x5)

        input_z = Input(shape=(shape_size,), name="latent_varaible")

        x = Reshape((z_dim[0], z_dim[1], layer_depth))(input_z)

        x = Conv2D(32, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        input_decode_1 = Input(
            (shape[0]//16, shape[1]//16, 64), name="decode_input_1")

        x = Concatenate(axis=-1)([x, input_decode_1])

        x = Conv2DTranspose(32, (5, 5), strides=(
            2, 2),  padding='same', use_bias=False)(x)

        input_decode_2 = Input(
            (shape[0]//8, shape[1]//8, 32), name="decode_input_2")

        x = Concatenate(axis=-1)([input_decode_2, x])

        x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(32, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2DTranspose(32, (5, 5), strides=(
            2, 2),  padding='same', use_bias=False)(x)

        input_decode_3 = Input(
            (shape[0]//4, shape[1]//4, 32), name="decode_input_3")

        x = Concatenate(axis=-1)([input_decode_3, x])

        x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(32, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2DTranspose(32, (5, 5), strides=(
            2, 2),  padding='same', use_bias=False)(x)

        input_decode_4 = Input(
            (shape[0]//2, shape[1]//2, 16), name="decode_input_4")

        x = Concatenate(axis=-1)([input_decode_4, x])

        x = Conv2D(16, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(16, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(16, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2DTranspose(16, (5, 5), strides=(
            2, 2),  padding='same', use_bias=False)(x)

        input_decode_5 = Input(
            (shape[0], shape[1], 8), name="decode_input_5")

        x = Concatenate(axis=-1)([input_decode_5, x])

        x = Conv2D(8, (3, 3),  activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(8, (3, 3),   activation=LeakyReLU(alpha=alpha), padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        x = Conv2D(1, (3, 3),  activation='tanh', padding='same',
                   kernel_regularizer=self.regularization, bias_regularizer=self.regularization)(x)

        self.encode_model = Model(
            inputs=input_img, outputs=[x1, x2, x3, x4, x5_cnn])

        self.decode_model = Model(
            inputs=[input_decode_5, input_decode_4, input_decode_3, input_decode_2, input_decode_1, input_z], outputs=x)

        self.encode_model.summary()
        self.decode_model.summary()

    def mapToDepth(self, A, P):

        shape = tf.shape(P)

        A = tf.reshape(A, shape=[shape[0], 1, 1, 1])

        return tf.math.divide_no_nan(A*(tf.ones_like(P)-P), P)

    @tf.function(experimental_compile=False)
    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mu

    @tf.function(experimental_compile=False)
    def __call__(self, image, z, training=False):

        q, mu, logvar = self.encode(image)

        return self.decode(q, z), mu, logvar

    @tf.function(experimental_compile=False)
    def encode(self, x):

        #mu, logvar, q = self.encode_model(x)

        out = self.encode_model(x)

        shape = tf.shape(out[0])

        mu = tf.zeros((shape[0], self.shape_size))

        logvar = tf.zeros((shape[0], self.shape_size))

        return out, mu, logvar

    @tf.function(experimental_compile=False)
    def decode(self, e, z):
        D_dist = self.decode_model([*e, z])

        D_dist = 0.5*(D_dist + tf.ones_like(D_dist))

        return D_dist

    def getTrainableParameters(self):
        return self.getTrainableVariables()

    def getTrainableVariables(self):
        return [*self.decode_model.trainable_variables,
                *self.encode_model.trainable_variables]

    def getLosses(self):
        return [*self.decode_model.losses, *self.encode_model.losses]

    @tf.function(experimental_compile=True)
    def sum_losses(self):
        loss = tf.add_n(self.decode_model.losses)
        loss += tf.add_n(self.encode_model.losses)

        return loss

    def getName(self):
        return "ShapeNetwork"

    def save(self, folder):
        path = os.path.join(folder, self.getName()+"_encode"+".hdf5")

        self.encode_model.save_weights(path)

        path = os.path.join(folder, self.getName()+"_decode"+".hdf5")

        self.decode_model.save_weights(path)

    def load(self, folder):

        loaded = True

        path = os.path.join(folder, self.getName()+"_encode"+".hdf5")

        if os.path.exists(path):
            self.encode_model.load_weights(path)
        else:
            loaded = False

        path = os.path.join(folder, self.getName()+"_decode"+".hdf5")

        if os.path.exists(path):
            self.decode_model.load_weights(path)
        else:
            loaded = False

        return loaded

    def getEncoder(self):
        return self.encode_model

    def getDecoder(self):
        return self.decode_model

    def print_model(self, dot_img_file):

        if os.path.exists(dot_img_file):
            shutil.rmtree(dot_img_file)

        os.makedirs(dot_img_file)
        """
        tf.keras.utils.plot_model(
            self.decode_model, to_file=dot_img_file+"decode.jpg", show_shapes=True, show_layer_names=False, expand_nested=True)
        tf.keras.utils.plot_model(
            self.encode_model, to_file=dot_img_file+"encode.jpg", show_shapes=True, show_layer_names=False, expand_nested=True)
        """
