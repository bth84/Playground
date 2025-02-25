from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Lambda, Input, Dense
from keras.losses import binary_crossentropy
from keras.utils import plot_model

import argparse
import matplotlib.pyplot as plt
import os

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#Get data, initialize usual suspects
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
original_dim = image_size**2
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#NN params
input_shape = (original_dim,)
intermediate_dim = 2**9  #512
batch_size = 128
latent_dim = 2
epochs = 50

#VAE model = encoder + decoder
#encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
x2 = Dense(intermediate_dim // 2, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x2)
z_log_var = Dense(latent_dim, name='z_log_var')(x2)

#reparameterization trick (to push the sampling out as input)
#'output_shape' is not necessary with TD backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

#instantiate Encoder
encoder = Model(inputs, [z_mean,z_log_var, z], name = 'Encoder')
encoder.summary()
plot_model(encoder, to_file='/Users/Ben/Desktop/vae_encoder.png', show_shapes=True)

#decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x2 = Dense(intermediate_dim // 2, activation='relu')(latent_inputs)
x = Dense(intermediate_dim, activation='relu')(x2)
outputs = Dense(original_dim, activation='sigmoid')(x)

#instantiate Decoder
decoder = Model(latent_inputs, outputs, name='Decoder')
decoder.summary()
plot_model(decoder, to_file='/Users/Ben/Desktop/vae_decoder.png', show_shapes=True)

#instantiate VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = 'load H5 model trained weights'
    parser.add_argument('-w', '--weights', help=help_)
    help_ = 'Use mse loss instead of binary cross entropy (default)'
    parser.add_argument('-m', '--mse', help = help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # #VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=1)
    kl_loss *= -.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    if args.weights:
        vae.load_weights(args.weights)
    else:
        vae.fit(
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None)
        )
        vae.save_weights('vae_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae")




