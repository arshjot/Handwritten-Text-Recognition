import pickle
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        with open('../data/iam_h' + str(config.im_height) + '_w' + str(config.im_width) + '.pickle', 'rb') as f:
            data = pickle.load(f)
        train, val = data['train'], data['validation']
        self.train_label, self.val_label = train['labels'], val['labels']
        self.char_map_inv = {i: j for j, i in data['char_map'].items()}

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (train['images'], train['im_widths'], train['lab_lengths']))
        train_label_dataset = tf.data.Dataset.from_generator(
            self.train_generator, output_types=tf.int32, output_shapes=(tf.TensorShape([None])))
        self.train_dataset = tf.data.Dataset.zip((self.train_dataset, train_label_dataset))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=500, seed=42)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((val['images'], val['im_widths'], val['lab_lengths']))
        val_label_dataset = tf.data.Dataset.from_generator(
            self.val_generator, output_types=tf.int32, output_shapes=(tf.TensorShape([None])))
        self.val_dataset = tf.data.Dataset.zip((self.val_dataset, val_label_dataset))

        self.train_dataset = self.train_dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=((tf.TensorShape([self.config.im_height, self.config.im_width]),
                            tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([None])),
            padding_values=((tf.constant(0.0), tf.constant(0), tf.constant(0)), tf.constant(-1)))
        self.train_dataset = self.train_dataset.prefetch(tf.contrib.data.AUTOTUNE)

        self.val_dataset = self.val_dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=((tf.TensorShape([self.config.im_height, self.config.im_width]),
                            tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([None])),
            padding_values=((tf.constant(0.0), tf.constant(0), tf.constant(0)), tf.constant(-1)))
        self.val_dataset = self.val_dataset.prefetch(tf.contrib.data.AUTOTUNE)

        self.iterator = tf.data.Iterator.from_structure(
            self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(self.val_dataset)

        self.num_classes = data['num_chars']
        self.num_iterations_train = len(train['images']) // self.config.batch_size
        self.num_iterations_val = len(val['images']) // self.config.batch_size

    def train_generator(self):
        for el in self.train_label:
            yield el

    def val_generator(self):
        for el in self.val_label:
            yield el

    def initialize(self, sess, is_train):
        if is_train:
            sess.run(self.training_init_op)
        else:
            sess.run(self.validation_init_op)

    def get_input(self):
        return self.iterator.get_next()


def main():
    class Config:
        im_height = 40
        im_width = 800
        batch_size = 1

    tf.reset_default_graph()
    sess = tf.Session()

    data_loader = DataGenerator(Config)
    x, y = data_loader.get_input()
    x_im, x_w, x_len = x

    data_loader.initialize(sess, is_train=True)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, is_train=False)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y.shape, out_y.dtype)


if __name__ == '__main__':
    main()
