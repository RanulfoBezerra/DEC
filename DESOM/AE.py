"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper function

@author Florent Forest
@version 2.0
"""

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils import plot_model

def mlp_autoencoder(encoder_dims, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric autoencoder model.

    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        The decoder is symmetric with encoder, so number of layers of the AE is 2*len(encoder_dims)-1
        act: activation of AE intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of AE layers
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    input_shape = (128, 128, 3)
    x = Input(shape=input_shape, name='input')
    # input_shape = self.img_shape
    filters = [32, 64, 128, 10]

    model = Sequential()
    pad3 = 'same'
    # if self.img_shape[0] % 8 == 0:
    #     pad3 = 'same'
    # else:
    # #     pad3 = 'valid'
    encoded = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1')(x)

    encoded = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(encoded)

    encoded = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(encoded)

    encoded = Flatten()(encoded)

    encoded = Dense(units=filters[3], name='embedding')(encoded)

    decoded = encoded

    decoded = Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu')(decoded)

    decoded = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(decoded)
    decoded = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(decoded)

    decoded = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(decoded)

    decoded = Conv2DTranspose(input_shape[2] , 5, strides=2, padding='same', name='deconv1')(decoded)
    # input_shape = (128,128,3)
    # model = Sequential()
    # model.add(
    #     Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    #
    # model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))
    #
    # model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    #
    # model.add(Flatten())
    # model.add(Dense(units=filters[3], name='embedding'))
    # model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu'))
    #
    # model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    # model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    #
    # model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
    #
    # model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    #
    # layers = len(model.layers)
    # encoderLayer = int(layers / 2) - 1
    # print("===Layers", layers, encoderLayer)
    # autoencoder = model
    #
    # plot_model(model,to_file='AE.png')

    # optimizer = Adam(lr=0.001)
    #
    # model.compile(loss='mse', optimizer=optimizer)



    # encoder = Model(model.get_layer('deconv3').input, model.layers[encoderLayer].output, name='encoder')
    #
    # decoded = Model(encoder.output, model.layers[layers - 1].output, name='decoder')

    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    encoded_input = Input(shape=(10,))
    decoded = encoded_input
    decoded = Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu')(decoded)

    decoded = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(decoded)
    decoded = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(decoded)

    decoded = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(decoded)

    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(decoded)


    decoded = Model(inputs=encoded_input, outputs=decoded, name='decoder')


    return autoencoder, encoder, decoded

    # n_stacks = len(encoder_dims) - 1
    #
    # # Input
    # x = Input(shape=(encoder_dims[0],), name='input')
    # # Internal layers in encoder
    # encoded = x
    # for i in range(n_stacks-1):
    #     encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
    # # Hidden layer (latent space)
    # encoded = Dense(encoder_dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(encoded) # hidden layer, latent representation is extracted from here
    # # Internal layers in decoder
    # decoded = encoded
    # for i in range(n_stacks-1, 0, -1):
    #     decoded = Dense(encoder_dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(decoded)
    # # Output
    # decoded = Dense(encoder_dims[0], kernel_initializer=init, name='decoder_0')(decoded)
    #
    # # AE model
    # autoencoder = Model(inputs=x, outputs=decoded, name='AE')
    #
    # # Encoder model
    # encoder = Model(inputs=x, outputs=encoded, name='encoder')
    #
    # # Create input for decoder model
    # encoded_input = Input(shape=(encoder_dims[-1],))
    # # Internal layers in decoder
    # decoded = encoded_input
    # for i in range(n_stacks-1, -1, -1):
    #     decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # # Decoder model
    # decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    #
    # return autoencoder, encoder, decoder
