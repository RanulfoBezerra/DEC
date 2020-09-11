"""
Keras implementation for Convolutional Deep Embedded Clustering (DEC) algorithm:

        Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

Usage:
    use `python DEC.py -h` for help.

Author:
    Ranulfo Bezerra 2020 ; Xifeng Guo. 2017.1.30
"""

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
from tensorflow.keras.layers import concatenate

import sys
import glob
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from keras.models import load_model
import math
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        self.encoder = load_model('/media/ranulfo/Data/DEC/models/encoderModel_best.hdf5')

        n_data = 4

        model = Sequential()
        model.add(Dense(4, input_dim=self.input_dim, activation="relu"))
        model.add(Dense(2, activation="linear"))

        self.mlp_model = model

        input2 = self.mlp_model.input
        input1 = self.encoder.output

        cluster_input = concatenate([input1, input2])


        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(input1)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def pretrain_mlp(self, x, y=None, optimizer='adam', epochs=200, batch_size=10, save_dir='results/temp'):
        print('...Pretraining...')
        self.mlp_model.compile(optimizer=optimizer, loss='mse')

        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs/10) != 0 and epoch % int(epochs/10) != 0:
                        return
                    feature_model = Model(self.model.input,
                                          self.model.get_layer(
                                              'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' '*8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.mlp_model.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.mlp_model.save_weights(save_dir + '/mlp_weights.h5')
        self.mlp_model.save(save_dir + '/mlp.h5')
        print('Pretrained weights are saved to %s/mlp_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def silluete_n_clusters(self, x):
        # kmean_input = np.concatenate((x[1], self.encoder.predict(x[0])), axis=1)
        kmean_input = self.encoder.predict(x)
        kmax = 158
        sil = []
        print('Initialize KMeans')
        sse = []
        points = kmean_input
        past_sse = 999
        min_sse_dif = 9999
        for k in range(65, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

            if abs(past_sse - curr_sse) < min_sse_dif:
                min_sse_dif = abs(past_sse - curr_sse)
                index = k
                print("Int Label", k, min_sse_dif)

            past_sse = curr_sse

            sse.append(curr_sse)
        print("Label", index)
        self.n_clusters = index

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):

        from sklearn.metrics import silhouette_score

        print('Update interval', update_interval)
        save_interval = int(x[0].shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        # kmean_input = np.concatenate((x[1], self.encoder.predict(x[0])), axis=1)
        kmean_input =  self.encoder.predict(x[0])
        # encoded_data = self.encoder.predict(x)
        #
        # data = np.reshape(encoded_data, (8,8))


        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

        # for k in range(65, kmax + 1):
        #     kmeans = KMeans(n_clusters=k).fit(kmean_input)
        #     labels = kmeans.labels_
        #     sil.append(silhouette_score(kmean_input, labels, metric='euclidean'))
        # plt.plot(sse)
        # plt.show()

        y_pred = kmeans.fit_predict(kmean_input)
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x[0].shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x[0], verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x[0].shape[0])]
            loss = self.model.train_on_batch(x=x[0][idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x[0].shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('Iter %d: loss = %.5f, delta = %.5f' % (ite, loss, delta_label)
                      )
                # print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_mlp_d_scc.h5')
        self.model.save_weights(save_dir + '/DEC_model_mlp_d_scc.h5')

        return y_pred


import keras.backend as K


def custom_mse(y_true, y_pred):
    # calculating squared difference between target and predicted values
    loss = K.square(y_pred - y_true)  # (batch_size, 2)

    # multiplying the values with weights along batch dimension
    loss = loss * [0.3, 0.7]  # (batch_size, 2)

    # summing both loss values along batch dimension
    loss = K.sum(loss, axis=1)  # (batch_size,)

    return loss

def load_image(path, imgpath):
    dim = 128
    image_list = np.zeros((len(path), 128, 128, 3))
    extension = 'png'
    for i, fig in enumerate(path):
        # img = image.load_img(fig, target_size=(200, 200)) #color_mode='grayscale'
        # x = image.img_to_array(img).astype('float32')
        # x = x / 255.0

        image = cv2.imread(imgpath + str(i).zfill(5) + '.' + extension)
        if image is None:
            print(imgpath + str(i).zfill(5) + '.' + extension)
            continue
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (dim, dim))
        # gray_img = cv2.resize(gray_img, (128, 128))
        # cv2.imwrite("gray_images/gray_" + str(count) + ".jpg", gray_img)
        # cv2.imwrite("color_images/color_" + str(count) + ".jpg", image)
        x = np.asarray(image)
        x = x / 255.0
        image_list[i] = x

    return image_list


def load_data(path):
    data = np.load(path+'mapped_tracked.npy')
    data = np.transpose(data)
    x = data[0]
    y = data[1]
    rot = data[2]
    id = data[4]
    img = data[5]
    cam = data[6]

    idT = np.zeros(id.shape)

    for i in range(len(data[4])):
        idT[i] = int(str(int(cam[i])) + str(int(id[i])))

    idT_max = max(idT)


    x_min = min(x)

    y_min = min(y)
    img_max = max(img)

    if y_min < 0:
        y = y - y_min
    if x_min < 0:
        x = x - x_min

    x_max = max(x)
    y_max = max(y)
    rot = rot + math.pi
    rot = rot / (math.pi * 2.0)

    x = x / float(x_max)
    y = y / float(y_max)
    img = img / float(img_max)
    idT = idT / float(idT_max)
    data_f = [idT, img]
    data_f = np.transpose(data_f)
    return data_f

def silluete_n_clusters(x, encoder):
    # kmean_input = np.concatenate((x[1], encoder.predict(x[0])), axis=1)
    kmean_input = encoder.predict(x[0])
    kmax = 158
    threshold = 0.5
    sil = []
    print('Initialize KMeans')
    sse = []
    points = kmean_input
    past_sse = 999
    min_sse_dif = 9999
    for k in range(65, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        dist = abs(past_sse - curr_sse)




        if dist < min_sse_dif:
            if dist < threshold:
                index = k
                break
            min_sse_dif = dist
            index = k
            print("Int Label", k, min_sse_dif)

        past_sse = curr_sse

        sse.append(curr_sse)
    print("Label", index)
    return  index



if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'fmnist', 'usps', 'reuters10k', 'stl'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=None, type=int)
    parser.add_argument('--update_interval', default=None, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    imgpath = '/media/ranulfo/Data/DEC/car_index/'
    TRAIN_IMAGES = glob.glob('/media/ranulfo/Data/DEC/car_index/*.png')
    TEST_IMAGES = glob.glob('data/cars/test/*.png')



    x_img = load_image(TRAIN_IMAGES, imgpath)
    x_test = load_image(TRAIN_IMAGES, imgpath)

    x_data = load_data('/media/ranulfo/Data/DEC/')


    a = open('/media/ranulfo/Data/DEC/data.txt', 'w')
    for data in x_data:
        a.write(str(data) + '\n')
    a.close()

    # # load dataset
    # from datasets import load_data
    # x, y = load_data(args.dataset)
    n_clusters = 70 # len(np.unique(y))

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    # setting parameters
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        update_interval = 140
        pretrain_epochs = 300
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
    elif args.dataset == 'reuters10k':
        update_interval = 30
        pretrain_epochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
    elif args.dataset == 'usps':
        update_interval = 30
        pretrain_epochs = 50
    elif args.dataset == 'stl':
        update_interval = 30
        pretrain_epochs = 10
    elif args.dataset == 'cars':
        update_interval = 30
        pretrain_epochs = 10

    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.pretrain_epochs is not None:
        pretrain_epochs = args.pretrain_epochs

    encoder = load_model('/media/ranulfo/Data/DEC/models/encoderModel_best.hdf5')

    # n_clusters = silluete_n_clusters([x_img, x_data], encoder)
    n_clusters = 158

    # prepare the DEC model
    dec = DEC(dims=[x_data.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)

    # if args.ae_weights is None:
    # dec.pretrain_mlp(x=x_data, y=None, optimizer=pretrain_optimizer,
    #              epochs=pretrain_epochs, batch_size=256,
    #              save_dir=args.save_dir)
    # else:
    #     dec.autoencoder.load_weights(args.ae_weights)



    dec.model.summary()
    t0 = time()
    dec.compile(optimizer=SGD(0.01, 0.9), loss='kld') #sparse_categorical_crossentropy kld
    y_pred = dec.fit([x_img,x_data], y=None, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=update_interval, save_dir=args.save_dir)


    # print('acc:', metrics.acc(y, y_pred))
    print('clustering time: ', (time() - t0))

    # dec.model.load_weights('/home/ranulfo/Projects/Python/DEC/results/DEC_model_mlp_d_scc_2.h5')
    y_pred = dec.model.predict([x_img,x_data])
    print(y_pred)
    print(y_pred.shape)

    for i, img in enumerate(x_img):
        img = img * 255
        ind = np.argmax(y_pred[i])
        cv2.imwrite('/media/ranulfo/Data/DEC/data/cars/result_img_158/' +str(ind) + '_'+ str(i) + '.png', img)

