import sys

sys.path.extend(['..'])

import pickle
import tensorflow as tf
from utils.augment import Augmentor
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # Load data
        with open('../data/iam_h' + str(config.im_height) + '_char_map.pkl', 'rb') as f:
            data = pickle.load(f)
        char_map = data['char_map']
        self.char_map_inv = {i: j for j, i in char_map.items()}
        self.train_dataset = tf.data.TFRecordDataset('../data/iam_h' + str(config.im_height) + '_train.tfrecords')
        self.val_dataset = tf.data.TFRecordDataset('../data/iam_h' + str(config.im_height) + '_val.tfrecords')
        self.test_dataset = tf.data.TFRecordDataset('../data/iam_h' + str(config.im_height) + '_test.tfrecords')

        # Parse, augment (if training set) and batch
        padded_shapes = ((tf.TensorShape([self.config.im_height, None])),
                         (tf.TensorShape([None])), (tf.TensorShape([])), (tf.TensorShape([])))
        padding_values = ((tf.constant(0.0)), (tf.constant(0)), (tf.constant(0)), (tf.constant(-1)))
        self.train_dataset = self.train_dataset.map(lambda x: self.parser(x, True),
                                                    num_parallel_calls=self.config.batch_size)\
            .shuffle(buffer_size=500)\
            .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        self.val_dataset = self.val_dataset.map(self.parser, num_parallel_calls=self.config.batch_size)\
            .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        self.test_dataset = self.test_dataset.map(self.parser, num_parallel_calls=self.config.batch_size)\
            .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        # Batch
        self.iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(self.val_dataset)

        self.num_classes = data['num_chars']
        self.num_iterations_train = data['len_train'] // self.config.batch_size
        self.num_iterations_val = data['len_val'] // self.config.batch_size

    def parser(self, record, do_augment=False):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string),
            'label': tf.VarLenFeature(tf.int64),
            'width': tf.FixedLenFeature((), tf.int64),
            'lab_length': tf.FixedLenFeature((), tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.image.decode_png(parsed['image_raw'], 1, tf.uint8)
        label = tf.sparse.to_dense(tf.cast(parsed['label'], tf.int32))
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
        return image, label, width, lab_length

    def initialize(self, sess, is_train):
        if is_train:
            sess.run(self.training_init_op)
        else:
            sess.run(self.validation_init_op)

    def get_input(self):
        return self.iterator.get_next()


def main():
    class Config:
        im_height = 128
        batch_size = 1

    tf.reset_default_graph()
    sess = tf.Session()

    data_loader = DataGenerator(Config)
    x_im, y, x_w, x_len = data_loader.get_input()

    data_loader.initialize(sess, is_train=True)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])

    with open('../data/iam_h' + str(Config.im_height) + '_char_map.pkl', 'rb') as f:
        data = pickle.load(f)
    char_map = data['char_map']
    char_map_inv = {i: j for j, i in char_map.items()}
    print(''.join([char_map_inv[i] for i in out_y[0]]))
    plt.imshow(out_x_im.reshape(128, -1), cmap='gray')
    plt.show()

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, is_train=False)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])

    print(''.join([char_map_inv[i] for i in out_y[0]]))
    plt.imshow(out_x_im.reshape(128, -1), cmap='gray')
    plt.show()

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y.shape, out_y.dtype)


if __name__ == '__main__':
    main()
