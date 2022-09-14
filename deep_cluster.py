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
            cluster_epsilon,
            pretrained_model_path='./encoder.h5'):
        super().__init__(input_shape, encoding_dim)
        self.pool = ThreadPoolExecutor(8)
        self.train_image_paths = glob(rf'{train_image_path}/**/*.jpg', recursive=True)
        self.len_train_image_paths = len(self.train_image_paths)
        self.cluster_output_save_path = cluster_output_save_path
        self.encoding_dim = encoding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_cluster_classes = num_cluster_classes
        self.cluster_epsilon = cluster_epsilon
        self.img_type = cv2.IMREAD_COLOR
        self.pretrained_model_path = pretrained_model_path
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
        def graph_forward(model, x):
            return model(x, training=False)
        if use_saved_model:
            if os.path.exists(self.pretrained_model_path) and os.path.isfile(self.pretrained_model_path):
                self.encoder = tf.keras.models.load_model('encoder.h5', compile=False)
                self.encoding_dim = np.asarray(self.encoder.output_shape).reshape(-1)[-1]
            else:
                print(f'pretrained model not found : [{self.pretrained_model_path}]')
        else:
            def training_view(batch, logs):
                global live_view_previous_time
                cur_time = time()
                if cur_time - live_view_previous_time > 0.5:
                    live_view_previous_time = cur_time
                    __x = cv2.imread(self.train_image_paths[randrange(0, self.len_train_image_paths)], self.img_type)
                    __x = cv2.resize(__x, (self.input_shape[1], self.input_shape[0]))
                    __x = np.asarray(__x).reshape((1,) + self.input_shape) / 255.0
                    __y = graph_forward(self.ae, __x)
                    __x = np.asarray(__x) * 255.0
                    __x = np.clip(__x, 0, 255).astype('uint8').reshape(self.input_shape)
                    __y = np.asarray(__y) * 255.0
                    __y = np.clip(__y, 0, 255).astype('uint8').reshape(self.input_shape)
                    __x = cv2.resize(__x, (128, 128), interpolation=cv2.INTER_LINEAR)
                    __y = cv2.resize(__y, (128, 128), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('cae', np.concatenate((__x, __y), axis=1))
                    cv2.waitKey(1)

            self.ae.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss=tf.keras.losses.MeanSquaredError())
            # self.ae.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss=tf.keras.losses.BinaryCrossentropy())
            self.ae.summary()
            print(f'\ntrain on {self.len_train_image_paths} samples\n')
            self.ae.fit(
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
            latent_vectors.append(np.asarray(graph_forward(self.encoder, f.result())).reshape((self.encoding_dim,)))
        latent_vectors = np.asarray(latent_vectors)

        print('\nstart clustering. please wait...\n')
        criteria = (cv2.TERM_CRITERIA_EPS, -1, self.cluster_epsilon)
        compactness, labels, _ = cv2.kmeans(latent_vectors, self.num_cluster_classes, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)
        average_distance_from_each_label_center = compactness / float(len(self.train_image_paths))
        print(f'clustering success. average distance from each label center : {average_distance_from_each_label_center:.4f}')

        print('\nsaving data\n')
        fs = []
        for i in range(self.len_train_image_paths):
            fs.append(self.pool.submit(cv2.imread, self.train_image_paths[i], self.img_type))
        for i in tqdm(range(self.len_train_image_paths)):
            img = fs[i].result()
            save_dir = rf'{self.cluster_output_save_path}\class_{str(labels[i]).replace("[", "").replace("]", "")}'
            if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
                os.makedirs(save_dir, exist_ok=True)
            basename = os.path.basename(self.train_image_paths[i])
            cv2.imwrite(rf'{save_dir}\{basename}', img)

    def evaluate_clustered_data_using_label(self, num_classes):
        data_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        for path in glob(f'{self.cluster_output_save_path}/**/*.jpg', recursive=True):
            basename = os.path.basename(path)
            class_index = int(basename.split('_')[0])
            data_counts[class_index] += 1

        clustered_datas = []
        for dir_path in glob(f'{self.cluster_output_save_path}/*'):
            dir_path = dir_path.replace('\\', '/')
            if os.path.isdir(dir_path) and dir_path.find('class_') > -1:
                clustered_datas.append({
                    'class_name': os.path.basename(dir_path),
                    'class_index': -1,
                    'clustered_accuracy': 0.0,
                    'clustered_data_counts': np.zeros(shape=(num_classes,), dtype=np.int32)})
                for image_path in glob(f'{dir_path}/*.jpg'):
                    image_path = image_path.replace('\\', '/')
                    basename = os.path.basename(image_path)
                    class_index = int(basename.split('_')[0])
                    clustered_datas[-1]['clustered_data_counts'][class_index] += 1

        assert len(clustered_datas) == num_classes
        confirmed_class_indexes = []
        not_confirmed_class_indexes = [i for i in range(num_classes)]

        for i in range(num_classes):
            max_accuracy = 0.0
            max_accuracy_index = -1
            clustered_data_index = -1
            for j in range(num_classes):
                if clustered_datas[j]['class_index'] == -1:
                    clustered_class_index = np.argmax(clustered_datas[j]['clustered_data_counts'])
                    if clustered_class_index in confirmed_class_indexes:
                        clustered_data_counts_copy = clustered_datas[j]['clustered_data_counts'].tolist()
                        for _ in range(len(clustered_data_counts_copy)):
                            clustered_class_index = np.argmax(clustered_data_counts_copy)
                            if clustered_class_index in confirmed_class_indexes:
                                clustered_data_counts_copy[clustered_class_index] = 0
                            else:
                                break
                    if clustered_class_index in not_confirmed_class_indexes:
                        clustered_accuracy = float(clustered_datas[j]['clustered_data_counts'][clustered_class_index]) / float(data_counts[clustered_class_index])
                        if clustered_accuracy > max_accuracy:
                            max_accuracy = clustered_accuracy
                            max_accuracy_index = clustered_class_index
                            clustered_data_index = j
            if clustered_data_index > -1:
                clustered_datas[clustered_data_index]['clustered_accuracy'] = max_accuracy
                clustered_datas[clustered_data_index]['class_index'] = max_accuracy_index
                confirmed_class_indexes.append(max_accuracy_index)
                not_confirmed_class_indexes.remove(max_accuracy_index)

        # assign not clustered class index
        not_clustered_count = 0
        for i in range(num_classes):
            if clustered_datas[i]['class_index'] == -1:
                not_clustered_count += 1
        assert not_clustered_count == len(not_confirmed_class_indexes)
        inc = 0
        for i in range(num_classes):
            if clustered_datas[i]['class_index'] == -1:
                clustered_datas[i]['class_index'] = not_confirmed_class_indexes[inc]
                inc += 1

        acc_sum = 0.0
        clustered_datas = sorted(clustered_datas, key=lambda x: x['clustered_accuracy'], reverse=True)
        print()
        for i in range(num_classes):
            class_name = clustered_datas[i]['class_name']
            class_index = clustered_datas[i]['class_index']
            accuracy = clustered_datas[i]['clustered_accuracy']
            clustered_data_count = clustered_datas[i]['clustered_data_counts'][class_index]
            real_data_count = data_counts[class_index]
            acc_sum += accuracy
            print(f'dir_path : {class_name:10s}, class index : {class_index:3d}, clustered data count : {clustered_data_count:6d}, real data count : {real_data_count:6d}, accuracy : {accuracy:.4f}')
        average_accuracy = acc_sum / float(num_classes)
        print(f'average accuracy : {average_accuracy:.4f}')
