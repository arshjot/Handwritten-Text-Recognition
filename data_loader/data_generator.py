import pickle
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        with open('../data/iam_h' + str(config.im_height) + '_w' + str(config.im_width) + '.pickle', 'rb') as f:
            data = pickle.load(f)
        train, val = data['train'], data['validation']

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (train['images'], train['im_widths'], train['lab_lengths']))
        train_label_dataset = tf.data.Dataset.from_sparse_tensor_slices(
            tf.SparseTensor(train['labels'][0], train['labels'][1], train['labels'][2]))
        self.train_dataset = tf.data.Dataset.zip((self.train_dataset, train_label_dataset))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=500, seed=42)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((val['images'], val['im_widths'], val['lab_lengths']))
        val_label_dataset = tf.data.Dataset.from_sparse_tensor_slices(
            tf.SparseTensor(val['labels'][0], val['labels'][1], val['labels'][2]))
        self.val_dataset = tf.data.Dataset.zip((self.val_dataset, val_label_dataset))

        self.iterator = tf.data.Iterator.from_structure(
            ((tf.float32, tf.int16, tf.int16), (tf.int64, tf.int32, tf.int64)), self.train_dataset.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(self.val_dataset)

    def initialize(self, sess, is_train):
        if is_train:
            sess.run(self.training_init_op)
        else:
            sess.run(self.validation_init_op)

    def get_input(self):
        return self.iterator.get_next()


def main():
    class Config:
        im_height = 32
        im_width = 320

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
    print(out_y[0].shape, out_y[0].dtype)

    data_loader.initialize(sess, is_train=False)
    out_x_im, out_x_w, out_x_len, out_y = sess.run([x_im, x_w, x_len, y])

    print(out_x_im.shape, out_x_im.dtype)
    print(out_x_w.shape, out_x_w.dtype)
    print(out_x_len.shape, out_x_len.dtype)
    print(out_y[0].shape, out_y[0].dtype)


if __name__ == '__main__':
    main()
