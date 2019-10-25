import sys

sys.path.extend(['..'])

import glob
import pickle
import tensorflow as tf
from utils.augment import Augmentor
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, config, eval_phase=False, eval_on_test_data=False):
        self.eval_phase = eval_phase
        self.config = config
        data_dir = '../data/' + config.dataset + '_h' + str(config.im_height) + '_'
        # Load data
        with open(data_dir + 'char_map.pkl', 'rb') as f:
            data = pickle.load(f)
        char_map = data['char_map']
        self.char_map_inv = {i: j for j, i in char_map.items()}

        # Parse, augment (if training set) and batch
        padded_shapes = ((tf.TensorShape([self.config.im_height, None])),
                         (tf.TensorShape([None])), (tf.TensorShape([])), (tf.TensorShape([])))
        padding_values = ((tf.constant(0.0)), (tf.constant(-1)), (tf.constant(0)), (tf.constant(0)))

        self.num_classes = data['num_chars']

        if eval_phase:  # Only load evaluation dataset
            if eval_on_test_data:
                self.test_dataset = tf.data.TFRecordDataset(data_dir + 'test.tfrecords')
                self.test_dataset = self.test_dataset.map(self.parser, num_parallel_calls=8) \
                    .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
                self.iterator = tf.data.Iterator.from_structure(
                    self.test_dataset.output_types, self.test_dataset.output_shapes)
                self.test_init_op = self.iterator.make_initializer(self.test_dataset)

                self.num_iterations_val = data['len_test'] // self.config.batch_size + \
                    int(not data['len_test'] % self.config.batch_size == 0)
            else:  # Predict on new images kept in 'samples' directory
                file_list = sorted(glob.glob('../samples/processed/*'))  # Sort to keep track of file names
                self.test_dataset = tf.data.Dataset.from_tensor_slices(file_list)
                self.test_dataset = self.test_dataset.map(self.parser_directory, num_parallel_calls=8) \
                    .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
                self.iterator = tf.data.Iterator.from_structure(
                    self.test_dataset.output_types, self.test_dataset.output_shapes)
                self.test_init_op = self.iterator.make_initializer(self.test_dataset)

                self.num_iterations_val = (len(file_list) // self.config.batch_size) + \
                    int(not len(file_list) % self.config.batch_size == 0)

        else:  # For training and validation sets
            self.train_dataset = tf.data.TFRecordDataset(data_dir + 'train.tfrecords')
            self.val_dataset = tf.data.TFRecordDataset(data_dir + 'val.tfrecords')

            if self.config.bucketing:
                self.train_dataset = self.train_dataset.map(lambda x: self.parser(x, True),
                                                            num_parallel_calls=8) \
                    .apply(tf.data.experimental.bucket_by_sequence_length(
                        element_length_func=self.element_length_fn,
                        bucket_boundaries=range(500, 4501, 200),
                        bucket_batch_sizes=[self.config.batch_size] * (len(range(500, 4501, 200)) + 1),
                        padded_shapes=padded_shapes,
                        padding_values=padding_values)) \
                    .shuffle(buffer_size=500)
            else:
                self.train_dataset = self.train_dataset.map(lambda x: self.parser(x, True),
                                                            num_parallel_calls=8) \
                    .shuffle(buffer_size=500) \
                    .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
            self.val_dataset = self.val_dataset.map(self.parser, num_parallel_calls=8) \
                .padded_batch(self.config.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

            # Batch
            self.iterator = tf.data.Iterator.from_structure(
                self.train_dataset.output_types, self.train_dataset.output_shapes)
            self.training_init_op = self.iterator.make_initializer(self.train_dataset)
            self.validation_init_op = self.iterator.make_initializer(self.val_dataset)

            self.num_iterations_train = data['len_train'] // self.config.batch_size + \
                int(not data['len_train'] % self.config.batch_size == 0)
            self.num_iterations_val = data['len_val'] // self.config.batch_size + \
                int(not data['len_val'] % self.config.batch_size == 0)

    def parser(self, record, do_augment=False):
        keys_to_features = {
            'image_raw': tf.FixedLenFeature((), tf.string),
            'label': tf.VarLenFeature(tf.int64),
            'width': tf.FixedLenFeature((), tf.int64),
            'lab_length': tf.FixedLenFeature((), tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.image.decode_image(parsed['image_raw'], 1, tf.uint8)
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

    def parser_directory(self, filename):
        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, 3, tf.uint8)
        image = tf.image.rgb_to_grayscale(image)
        width = tf.shape(image)[1]
        image = 1-tf.divide(tf.cast(image, tf.float32), 255.0)
        height = tf.constant(self.config.im_height, tf.int32)
        image = tf.reshape(image, [height, width])
        label, lab_length = [0], 0  # dummy values

        return image, label, width, lab_length

    def element_length_fn(self, im, lab, w, lab_len):
        return tf.shape(im)[1]

    def initialize(self, sess, is_train):
        if self.eval_phase:
            sess.run(self.test_init_op)
        elif is_train:
            sess.run(self.training_init_op)
        else:
            sess.run(self.validation_init_op)

    def get_input(self):
        return self.iterator.get_next()


def main():
    import numpy as np

    class Config:
        im_height = 128
        batch_size = 5
        dataset = "IAM"
        bucketing = 0

    tf.reset_default_graph()
    sess = tf.Session()

    data_loader = DataGenerator(Config)
    x_im, y, x_w, x_len = data_loader.get_input()
    y = tf.contrib.layers.dense_to_sparse(y, eos_token=-1).values
    x_im = tf.expand_dims(x_im, 3)
    # translation_vector = tf.cast(tf.stack([(tf.shape(x_im)[2] - x_w) / tf.constant(2),
    #                                        tf.constant(0.0, shape=[Config.batch_size], dtype=tf.float64)], axis=1),
    #                              tf.float32)
    # x_im = tf.contrib.image.translate(x_im, translation_vector)

    data_loader.initialize(sess, is_train=True)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])
    out_y = np.split(out_y, np.cumsum(out_x_len))
    out_y = out_y[0]
    if Config.batch_size > 1:
        out_x_im = out_x_im[0]

    with open('../data/'+Config.dataset+'_h' + str(Config.im_height) + '_char_map.pkl', 'rb') as f:
        data = pickle.load(f)
    char_map = data['char_map']
    char_map_inv = {i: j for j, i in char_map.items()}
    print(''.join([char_map_inv[i] for i in out_y]))
    plt.imshow(out_x_im.reshape(Config.im_height, -1), cmap='gray')
    plt.show()

    print(out_x_im.max(), out_x_im.dtype)
    print(out_x_w.shape, out_x_w)
    print(out_x_len.shape, out_x_len)
    print(out_y, out_y.dtype)

    data_loader.initialize(sess, is_train=False)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])
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
