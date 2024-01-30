"""
Copyright (c) 2022 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import warnings
import numpy as np
import shutil as sh
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler


class DeepCluster:
    def __init__(
            self,
            image_path,
            save_path,
            input_shape,
            lr,
            batch_size,
            iterations,
            latent_dim,
            n_cluster,
            eps=1e-7,
            use_kmeans_only=False,
            pretrained_model_path=''):
        self.image_path = image_path
        self.save_path = save_path
        self.input_shape = input_shape
        self.lr = lr
        self.batch_size = batch_size
        self.iterations = iterations
        self.latent_dim = latent_dim
        self.n_cluster = n_cluster
        self.eps = eps
        self.use_kmeans_only = use_kmeans_only
        self.pretrained_model_path = pretrained_model_path

        warnings.filterwarnings(action='ignore')
        self.model = Model(input_shape=self.input_shape, latent_dim=self.latent_dim)
        if self.use_kmeans_only:
            self.latent_dim = np.prod(self.input_shape)
        else:
            pretrained_ae_e = None
            if self.pretrained_model_path != '':
                if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
                    pretrained_ae_e = tf.keras.models.load_model(pretrained_model_path, compile=False)
                else:
                    print(f'model not found : {pretrained_model_path}')
                    exit(0)
            self.ae, self.ae_e = self.model.build(ae_e=pretrained_ae_e)
            self.latent_dim = self.ae_e.output_shape[-1]

        self.image_paths = glob(rf'{image_path}/**/*.jpg', recursive=True)
        self.data_generator = DataGenerator(
            image_paths=self.image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size)

    @staticmethod
    @tf.function
    def graph_forward(model, x):
        return model(x, training=False)

    @tf.function
    def compute_gradient(self, model, optimizer, x):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(x - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def print_loss(self, progress_str, loss):
        loss_str = f'\r{progress_str}'
        loss_str += f' loss : {loss:>8.4f}'
        print(loss_str, end='')

    def cluster(self):
        if len(self.image_paths) == 0:
            print(f'no images found in {self.image_path}')
            exit(0)

        if not self.use_kmeans_only:
            self.ae.summary()
            print(f'\ntrain on {len(self.image_paths)} samples\n')
            iteration_count = 0
            compute_gradient = tf.function(self.compute_gradient)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.0, policy='onecycle')
            eta_calculator = ETACalculator(iterations=self.iterations)
            eta_calculator.start()
            while True:
                batch_x = self.data_generator.load()
                lr_scheduler.update(optimizer, iteration_count)
                loss = compute_gradient(self.ae, optimizer, batch_x)
                iteration_count += 1
                progress_str = eta_calculator.update(iteration_count)
                self.print_loss(progress_str, loss)
                if iteration_count == self.iterations:
                    print('\ntrain end successfully')
                    break

        print('\nextracting latent vector')
        fs = []
        for i in range(len(self.image_paths)):
            fs.append(self.data_generator.pool.submit(self.data_generator.load_image, self.image_paths[i]))
        latent_vectors = []
        for f in tqdm(fs):
            img = f.result()
            x = self.data_generator.preprocess(img)
            if self.use_kmeans_only:
                latent_vectors.append(np.asarray(x).reshape((self.latent_dim,)))
            else:
                x = x.reshape((1,) + x.shape)
                latent_vectors.append(np.asarray(self.graph_forward(self.ae_e, x)).reshape((self.latent_dim,)))
        latent_vectors = np.asarray(latent_vectors)

        print('\nstart clustering. please wait...')
        criteria = (cv2.TERM_CRITERIA_EPS, -1, self.eps)
        sse, labels, _ = cv2.kmeans(latent_vectors, self.n_cluster, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        mse = sse / float(len(self.image_paths))
        print(f'clustering success. MSE : {mse:.4f}')

        print('\nsaving data')
        fs = []
        for i in range(len(self.image_paths)):
            fs.append(self.data_generator.pool.submit(self.data_generator.load_image, self.image_paths[i]))
        copy_paths = []
        for i in range(len(self.image_paths)):
            img = fs[i].result()
            class_index = str(labels[i]).replace("[", "").replace("]", "")
            save_dir = f'{self.save_path}/{class_index}'
            if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
                os.makedirs(save_dir, exist_ok=True)
            basename = os.path.basename(self.image_paths[i])
            copy_paths.append(f'{save_dir}/{basename}')
        fs = []
        for i in range(len(self.image_paths)):
            fs.append(self.data_generator.pool.submit(sh.copy, self.image_paths[i], copy_paths[i]))
        for f in tqdm(fs):
            f.result()
        print('\nclustering success')

