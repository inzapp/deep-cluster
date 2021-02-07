import os
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob
from random import randrange
from time import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from auto_encoder import AutoEncoder
from generator import DeepClusterDataGenerator

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class MeanAbsoluteLogError(tf.keras.losses.Loss):
    """
    Mean absolute logarithmic error loss function.
    f(x) = -log(1 - MAE(x))
    Usage:
     model.compile(loss=[MeanAbsoluteLogError()], optimizer="sgd")
    """

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = -tf.math.log(1.0 + 1e-7 - tf.math.abs(y_pred - y_true))
        return tf.keras.backend.mean(loss)


class DeepCluster(AutoEncoder):
    def __init__(
            self,
            train_image_path,
            cluster_output_save_path,
            input_shape,
            encoding_dim,
            lr,
            epochs,
            batch_size,
            num_cluster_classes,
            cluster_epsilon):
        super().__init__(input_shape, encoding_dim)
        self.pool = ThreadPoolExecutor(8)
        self.train_image_paths = glob(rf'{train_image_path}\*.jpg')
        self.len_train_image_paths = len(self.train_image_paths)
        self.cluster_output_save_path = cluster_output_save_path
        self.encoding_dim = encoding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_cluster_classes = num_cluster_classes
        self.cluster_epsilon = cluster_epsilon
        self.img_type = cv2.IMREAD_COLOR
        if self.input_shape[2] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE
        self.generator = DeepClusterDataGenerator(
            train_image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            encoding_dim=self.encoding_dim,
            batch_size=self.batch_size,
            img_type=self.img_type)

    def cluster(self, use_saved_model=False):

        @tf.function
        def predict_on_graph(model, x):
            return model(x, training=False)

        if use_saved_model:
            self.encoder = tf.keras.models.load_model('encoder.h5', compile=False)
        else:
            def training_view(batch, logs):
                global live_view_previous_time
                cur_time = time()
                if cur_time - live_view_previous_time > 0.5:
                    live_view_previous_time = cur_time
                    __x = cv2.imread(self.train_image_paths[randrange(0, self.len_train_image_paths)], self.img_type)
                    __x = cv2.resize(__x, (self.input_shape[1], self.input_shape[0]))
                    __x = np.asarray(__x).reshape((1,) + self.input_shape) / 255.0
                    __y = predict_on_graph(self.cae, __x)
                    __x = np.asarray(__x) * 255.0
                    __x = np.clip(__x, 0, 255).astype('uint8').reshape(self.input_shape)
                    __y = np.asarray(__y) * 255.0
                    __y = np.clip(__y, 0, 255).astype('uint8').reshape(self.input_shape)
                    __x = cv2.resize(__x, (128, 128), interpolation=cv2.INTER_LINEAR)
                    __y = cv2.resize(__y, (128, 128), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('cae', np.concatenate((__x, __y), axis=1))
                    cv2.waitKey(1)

            self.cae.compile(
                optimizer=tf.keras.optimizers.Adam(lr=self.lr),
                loss=MeanAbsoluteLogError())
            self.cae.summary()
            print(f'\ntrain on {self.len_train_image_paths} samples\n')
            self.cae.fit(
                x=self.generator,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=tf.keras.callbacks.LambdaCallback(on_batch_end=training_view))
            self.encoder.save('encoder.h5')
            cv2.destroyAllWindows()

        def load_x(img_path):
            __x = cv2.imread(img_path, self.img_type)
            __x = cv2.resize(__x, (self.input_shape[1], self.input_shape[0]))
            __x = np.asarray(__x).reshape((1,) + self.input_shape)
            return __x.astype('float32') / 255.0

        print('\nextracting latent vector\n')
        fs = []
        for i in range(self.len_train_image_paths):
            fs.append(self.pool.submit(load_x, self.train_image_paths[i]))
        latent_vectors = []
        for f in tqdm(fs):
            latent_vectors.append(np.asarray(predict_on_graph(self.encoder, f.result())).reshape((self.encoding_dim,)))
        latent_vectors = np.asarray(latent_vectors)

        print('\nstart clustering. please wait...\n')
        criteria = (cv2.TERM_CRITERIA_EPS, -1, self.cluster_epsilon)
        compactness, labels, _ = cv2.kmeans(latent_vectors, self.num_cluster_classes, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        print(f'clustering success. compactness : {compactness}')

        def load_img(img_path):
            return cv2.imread(img_path, self.img_type)

        print('\nsaving data\n')
        fs = []
        for i in range(self.len_train_image_paths):
            fs.append(self.pool.submit(load_img, self.train_image_paths[i]))
        for i in tqdm(range(self.len_train_image_paths)):
            img = fs[i].result()
            save_dir = rf'{self.cluster_output_save_path}\class_{str(labels[i]).replace("[", "").replace("]", "")}'
            if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
                os.makedirs(save_dir, exist_ok=True)
            file_name = self.train_image_paths[i].split('\\')[-1]
            cv2.imwrite(rf'{save_dir}\{file_name}', img)
