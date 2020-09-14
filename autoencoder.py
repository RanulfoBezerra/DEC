import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.models import load_model
from keras.callbacks import CSVLogger

from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model

TRAIN_IMAGES = glob.glob('/media/ranulfo/Data/DEC/data/cars/all/*.png')
TEST_IMAGES = glob.glob('/media/ranulfo/Data/DEC/data/cars/all/*.png')


def load_image(path):
    dim = 128
    image_list = np.zeros((len(path), dim, dim, 3))
    for i, fig in enumerate(path):
        # img = image.load_img(fig, target_size=(200, 200)) #color_mode='grayscale'
        # x = image.img_to_array(img).astype('float32')
        # x = x / 255.0
        image = cv2.imread(fig)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (dim, dim))
        # image = image.reshape((-1, 3))
        # gray_img = cv2.resize(gray_img, (128, 128))
        # cv2.imwrite("gray_images/gray_" + str(count) + ".jpg", gray_img)
        # cv2.imwrite("color_images/color_" + str(count) + ".jpg", image)
        x = np.asarray(image)
        x = x / 255.0
        image_list[i] = x

    return image_list


x_train = load_image(TRAIN_IMAGES)
x_test = load_image(TEST_IMAGES)


def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]

x_train, y_train, x_val, y_val = train_val_split(x_train, x_train)
print(x_train.shape, x_val.shape)


def save_imgs(imgs):
    for i, img in enumerate(imgs):
        img = img * 255
        cv2.imwrite('/media/ranulfo/Data/DEC/data/cars/ae/' + str(i) + '.png', img)
    print(imgs.shape)

def save_imgmatrix(encoder, path, num_clusters):
    # im = np.load(path+'img_data.npy')
    # print("img", im.shape, im)
    img_size_matrix = []
    track_path = path + 'Tracks-Img/'
    img_matrix = []
    img_matrix_mean = []
    initial_size = 0

    for cl_num in range(num_clusters):
        print(track_path+str(cl_num)+'_*.png')
        imgs_path = glob.glob(track_path+str(cl_num)+'_*.png')
        # print(imgs_path)
        img_list = load_image(imgs_path)
        imgs_data = encoder.predict(img_list)
        cluster_size, _ = imgs_data.shape
        cluster_size = initial_size + cluster_size + 1
        img_size_matrix.append([initial_size, cluster_size, len(imgs_path)])
        initial_size = cluster_size - 1
        img_matrix_mean.append(np.mean(imgs_data, axis=0))
        img_matrix.append(np.concatenate((np.mean(imgs_data, axis=0), np.std(imgs_data, axis=0) ) ) )

        if cl_num > 0:
            final_imgs_data = np.concatenate((final_imgs_data, imgs_data))
        else:
            final_imgs_data = imgs_data
    print(img_matrix)
    np.save(path+'img_data', final_imgs_data)
    np.save(path + 'img_data_mean', img_matrix_mean)
    np.save(path+'img_data_mean_std', img_matrix)
    np.save(path+'img_data_size',img_size_matrix)




class Autoencoder():
    def __init__(self):
        dim = 128
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.filepath = 'encoder.hdf5'

        optimizer = Adam(lr=0.001)

        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()

    def on_epoch_end(self, epoch, logs=None):
        # current = logs.get(self.monitor)
        # if self.monitor_op(current, self.best):
        #     self.best = current
            # self.encoder.save_weights(self.filepath, overwrite=True)
        self.encoder.save(self.filepath, overwrite=True)


    def build_model(self):
        input_shape = Input(shape=self.img_shape)
        input_shape = self.img_shape
        filters = [32, 64, 128, 10]
        model = Sequential()
        if self.img_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'
        model.add(
            Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=self.img_shape))

        model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

        model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

        model.add(Flatten())
        model.add(Dense(units=filters[3], name='embedding'))
        model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu'))

        model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
        model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

        model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

        model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
        model.summary()
        return model

    # def build_model(self):
    #     input_layer = Input(shape=self.img_shape)
    #
    #     print("IMAGE SHAPE===========", self.img_shape)
    #
    #     x = Conv2D(64, (3, 3), padding='same')(input_layer)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = MaxPooling2D((2, 2), padding='same')(x)
    #     x = Conv2D(32, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = MaxPooling2D((2, 2), padding='same')(x)
    #     x = Conv2D(16, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = MaxPooling2D((2, 2), padding='same')(x)
    #
    #     x = Conv2D(8, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     encoded = MaxPooling2D((2, 2), padding='same')(x)
    #
    #     # x = Conv2D(4, (3, 3), padding='same')(x)
    #     # x = BatchNormalization()(x)
    #     # x = Activation('relu')(x)
    #     # encoded = MaxPooling2D((2, 2), padding='same')(x)
    #
    #     # x = Conv2D(2, (3, 3), padding='same')(x)
    #     # x = BatchNormalization()(x)
    #     # x = Activation('relu')(x)
    #     # encoded = MaxPooling2D((2, 2), padding='same')(x)
    #
    #
    #
    #     # x = Conv2D(2, (3, 3), padding='same')(encoded)
    #     # x = BatchNormalization()(x)
    #     # x = Activation('relu')(x)
    #     # x = UpSampling2D((2, 2))(x)
    #
    #     # x = Conv2D(4, (3, 3), padding='same')(encoded)
    #     # x = BatchNormalization()(x)
    #     # x = Activation('relu')(x)
    #     # x = UpSampling2D((2, 2))(x)
    #
    #     x = Conv2D(8, (3, 3), padding='same')(encoded)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = UpSampling2D((2, 2))(x)
    #
    #     x = Conv2D(16, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = UpSampling2D((2, 2))(x)
    #     x = Conv2D(32, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = UpSampling2D((2, 2))(x)
    #     x = Conv2D(64, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Activation('relu')(x)
    #     x = UpSampling2D((2, 2))(x)
    #     x = Conv2D(3, (3, 3), padding='same')(x)
    #     x = BatchNormalization()(x)
    #     decoded = Activation('sigmoid')(x)
    #     output_layer = decoded
    #
    #     # # encoder
    #     # h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    #     # h = MaxPooling2D((2, 2), padding='same')(h)
    #     #
    #     # # decoder
    #     # h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    #     # h = UpSampling2D((2, 2))(h)
    #     # output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(h)
    #
    #     return Model(input_layer, output_layer)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=30):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
        csv_logger = CSVLogger('results/autoencoder_pretrain_log.csv')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[csv_logger])
        self.autoencoder_model.save("/media/ranulfo/Data/DEC/models/autoencoderModel_test.hdf5")
        self.autoencoder_model.save_weights("/media/ranulfo/Data/DEC/models/autoencoderWeights_test.hdf5", overwrite=True)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

# ae = Autoencoder()
# ae.train_model(x_train, y_train, x_val, y_val, epochs=300, batch_size=10)
# model = ae.autoencoder_model

# model = load_model('/media/ranulfo/Data/DEC/models/autoencoderModel_test.hdf5')
# score = model.evaluate(x_train, x_train, verbose=1)
# print("Score", score)
# layers = len(model.layers)
# encoderLayer = int(layers/2) - 1
# encoder = Model(model.layers[0].input, model.layers[encoderLayer].output)
# encoder.save("/media/ranulfo/Data/DEC/models/encoderModel_500.hdf5")
encoder = load_model("/media/ranulfo/RSSD/DEC/models/encoderModel_500.hdf5")
# model.compile(loss='mse', optimizer=optimizer)
encoder.summary()

track_path = '/media/ranulfo/RSSD/Data/map_traffic/Campus/results/4/'
save_imgmatrix(encoder, track_path, 159)

# print("Layers", len(model.layers))
# imgs = model.predict(x_test)
# datas = encoder.predict(x_test)
# print(datas.shape)

# save_imgs(imgs)
# save_imgs(x_test)


# for i, img in enumerate(imgs):
#     img = img * 255
#     cv2.imwrite('/media/ranulfo/Data/DEC/data/cars/ae/'+str(i)+'.png', img)
# print(imgs.shape)
#
# for i, img in enumerate(x_test):
#     img = img * 255
#     cv2.imwrite('/media/ranulfo/Data/DEC/data/cars/ae/'+str(i)+'_truth.png', img)
# print(imgs.shape)


# print(model.predict(x_test))


#
# input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format
#
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)
#
# # at this point the representation is (7, 7, 32)
#
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')