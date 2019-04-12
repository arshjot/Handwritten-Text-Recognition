import sys
sys.path.extend([sys.path[0]+'/..'])

import pickle
import tensorflow as tf
from utils.augment import Augmentor
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # Load data
        with open('./data/iam_h' + str(config.im_height) + '_char_map.pkl', 'rb') as f:
            data = pickle.load(f)
        char_map = data['char_map']
        self.char_map_inv = {i: j for j, i in char_map.items()}
        self.train_dataset = tf.data.TFRecordDataset('./data/iam_h' + str(config.im_height) + '_train.tfrecords')
        self.val_dataset = tf.data.TFRecordDataset('./data/iam_h' + str(config.im_height) + '_val.tfrecords')
        self.test_dataset = tf.data.TFRecordDataset('./data/iam_h' + str(config.im_height) + '_test.tfrecords')

        # Parse, augment (if training set) and batch
        self.train_dataset = self.train_dataset.map(lambda x: self.parser(x, False),
                                                    num_parallel_calls=self.config.batch_size)\
            .shuffle(buffer_size=200) \
            .window(self.config.batch_size) \
            .flat_map(self.batch_fn)
        self.val_dataset = self.val_dataset.map(self.parser, num_parallel_calls=self.config.batch_size) \
            .shuffle(buffer_size=50) \
            .window(self.config.batch_size) \
            .flat_map(self.batch_fn)
        self.test_dataset = self.test_dataset.map(self.parser, num_parallel_calls=self.config.batch_size) \
            .window(self.config.batch_size) \
            .flat_map(self.batch_fn)

        self.num_classes = data['num_chars']

    def parser(self, record, do_augment=False):
        keys_to_features = {
            'image_raw': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.VarLenFeature(tf.int64),
            'width': tf.io.FixedLenFeature((), tf.int64),
            'lab_length': tf.io.FixedLenFeature((), tf.int64),
        }
        parsed = tf.io.parse_single_example(serialized=record, features=keys_to_features)
        image = tf.image.decode_png(parsed['image_raw'], 1, tf.uint8)
        label = tf.cast(parsed['label'], tf.int32)
        lab_length = tf.cast(parsed['lab_length'], tf.int32)
        height = tf.constant(self.config.im_height, tf.int32)
        width = tf.cast(parsed['width'], tf.int32)
        image = tf.reshape(image, [height, width, tf.constant(1)])
        image = tf.cast(image, tf.float32)

        if do_augment:
            aug_img = Augmentor(image, height, width)
            aug_img.random_scaling(prob=0.5)
            aug_img.random_shearing(prob=0.5)
            aug_img.random_rotation(prob=0.5)
            aug_img.random_translation(prob=0.5)
            aug_img.random_dilation(prob=0.5)
            aug_img.random_erosion(prob=0.5)
            image = aug_img.image
            width = aug_img.width

        image = tf.reshape(image, [height, width])
        return image, width, lab_length, label

    def batch_fn(self, im_dense, w_dense, lab_len_dense, sparse):
        input_set = tf.data.Dataset.zip((
            im_dense.padded_batch(self.config.batch_size, padded_shapes=tf.TensorShape([self.config.im_height, None]),
                                  padding_values=tf.constant(0.0)),
            w_dense.batch(self.config.batch_size),
            lab_len_dense.batch(self.config.batch_size),
            sparse.batch(self.config.batch_size)))

        return tf.data.Dataset.zip((input_set, sparse.batch(self.config.batch_size)))


def main():
    import numpy as np

    class Config:
        im_height = 128
        batch_size = 5

    data_loader = DataGenerator(Config)
    for sample in data_loader.train_dataset:
        [out_x_im, out_x_w, out_x_len, out_y], _ = sample
        break

    out_y = out_y.values
    out_x_im = tf.expand_dims(out_x_im, 3)
    out_x_im, out_x_w, out_x_len, out_y = out_x_im.numpy(), out_x_w.numpy(), out_x_len.numpy(), out_y.numpy()

    out_y = np.split(out_y, np.cumsum(out_x_len))
    out_y = out_y[0]
    if Config.batch_size > 1:
        out_x_im = out_x_im[0]

    with open('./data/iam_h' + str(Config.im_height) + '_char_map.pkl', 'rb') as f:
        data = pickle.load(f)
    char_map = data['char_map']
    char_map_inv = {i: j for j, i in char_map.items()}
    print(''.join([char_map_inv[i] for i in out_y]))
    plt.imshow(out_x_im.reshape(Config.im_height, -1), cmap='gray')
    plt.show()

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y.shape, out_y.dtype)

    for sample in data_loader.val_dataset:
        [out_x_im, out_x_w, out_x_len, out_y], _ = sample
        break

    out_y = out_y.values
    out_x_im = tf.expand_dims(out_x_im, 3)
    out_x_im, out_x_w, out_x_len, out_y = out_x_im.numpy(), out_x_w.numpy(), out_x_len.numpy(), out_y.numpy()
    out_y = np.split(out_y, np.cumsum(out_x_len))
    out_y = out_y[0]
    if Config.batch_size > 1:
        out_x_im = out_x_im[0]

    print(''.join([char_map_inv[i] for i in out_y]))
    plt.imshow(out_x_im.reshape(Config.im_height, -1), cmap='gray')
    plt.show()

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y.shape, out_y.dtype)


if __name__ == '__main__':
    main()
