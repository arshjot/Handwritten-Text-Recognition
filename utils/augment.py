import tensorflow as tf
import numpy as np


class Augmentor:
    def __init__(self, image, height=None, width=None):
        self.image = image
        self.height = height
        self.width = width
        self.height_float = tf.cast(self.height, tf.float32)
        self.width_float = tf.cast(self.width, tf.float32)

    @staticmethod
    def transform(img_in, forward_transform, out_shape=None):
        t = tf.contrib.image.matrices_to_flat_transforms(tf.linalg.inv(forward_transform))
        img_out = tf.contrib.image.transform(img_in, t, interpolation="BILINEAR", output_shape=out_shape)
        return img_out

    @staticmethod
    def geom_prob(value, r):
        return ((1-r)**(value-3))*r

    @staticmethod
    def bernoulli_np_value(x, y, z, array_shape, r):
        mid_x, mid_y, mid_z = np.array([i//2 for i in array_shape])
        d_matrix = ((y - mid_y) ** 2 + (x - mid_x) ** 2 + (z - mid_z) ** 2) ** 0.5
        f = lambda d: np.random.choice(2, 1, p=[1 - np.exp(-d * r), np.exp(-d * r)])
        d_matrix = np.array([f(i) for i in d_matrix.flatten()])
        return d_matrix.reshape(array_shape)

    def get_kernel(self, srate, rrate):
        allowed_kernel_sizes = [3, 5, 7, 9, 11, 15]
        kernel_size_probs = np.array([self.geom_prob(k, srate) for k in allowed_kernel_sizes])
        kernel_size_probs /= kernel_size_probs.sum()
        kernel_sizes = np.random.choice(allowed_kernel_sizes, 2, p=kernel_size_probs)
        kernel_shape = (kernel_sizes[0], kernel_sizes[1], 1)
        return np.fromfunction(self.bernoulli_np_value, kernel_shape, dtype=np.float32,
                               array_shape=kernel_shape, r=rrate).astype(np.float32)

    def pad_and_rotate(self, height_pad, rot_factor):
        im = tf.image.pad_to_bounding_box(
            self.image, tf.cast(tf.math.divide(height_pad, tf.constant(2, tf.int32)), tf.int32), 0,
            tf.math.add(self.height, height_pad), self.width)
        im = tf.contrib.image.rotate(im, rot_factor)
        return tf.image.resize_images(im, (self.height, self.width))

    def random_scaling(self, stdv=0.12, prob=0.5):
        scale_factor = tf.cast(tf.py_func(
            lambda x: np.random.lognormal(mean=0, sigma=x), [stdv], tf.float64), tf.float32)
        apply_aug = tf.math.less(tf.random.uniform([]), tf.constant(prob))
        self.width = tf.cond(apply_aug,
                             lambda: tf.cast(tf.multiply(self.width_float, scale_factor), tf.int32), lambda: self.width)
        self.image = tf.cond(apply_aug,
                             lambda: self.transform(self.image, [[scale_factor, 0, 0],
                                                                 [0, scale_factor, 0],
                                                                 [0, 0, 1.0]], [self.height, self.width]),
                             lambda: self.image)

    def random_rotation(self, prec=100, prob=0.5):
        rot_factor = tf.reshape(tf.cast(tf.py_func(
            lambda x: np.random.vonmises(mu=0, kappa=x), [prec], tf.float64), tf.float32), [])
        height_pad = tf.cast(tf.math.scalar_mul(self.width_float, tf.math.sin(rot_factor)), tf.int32)
        height_pad = tf.cond(tf.math.less(rot_factor, tf.constant(0.0)), lambda: tf.abs(height_pad), lambda: height_pad)
        apply_aug = tf.math.less(tf.random.uniform([]), tf.constant(prob))
        self.image = tf.cond(apply_aug, lambda: self.pad_and_rotate(height_pad, rot_factor), lambda: self.image)

    def random_shearing(self, prec=4, prob=0.5):
        shear_factor = tf.cast(tf.py_func(
            lambda x: np.random.vonmises(mu=0, kappa=x), [prec], tf.float64), tf.float32)
        apply_aug = tf.math.less(tf.random.uniform([]), tf.constant(prob))
        self.image = tf.cond(apply_aug,
                             lambda: self.transform(self.image, [[1.0, shear_factor, 0],
                                                 [0, 1.0, 0],
                                                 [0, 0, 1.0]]), lambda: self.image)

    def random_translation(self, stdv=0.02, prob=0.5):
        trans_factor = tf.random.normal([2], mean=0, stddev=stdv)
        trans_factor_w = tf.reshape(tf.gather(trans_factor, [0]), [])
        trans_factor_h = tf.reshape(tf.gather(trans_factor, [1]), [])
        apply_aug = tf.math.less(tf.random.uniform([]), tf.constant(prob))
        self.image = tf.cond(apply_aug,
                             lambda: tf.contrib.image.translate(self.image, [trans_factor_w*self.width_float,
                                                                             trans_factor_h*self.height_float]),
                             lambda: self.image)

    def random_erosion(self, srate=0.8, rrate=1.2, prob=0.5):
        kernel = tf.py_func(self.get_kernel, [srate, rrate], tf.float32)
        apply_aug = tf.math.less(tf.random.uniform([]), tf.constant(prob))
        self.image = tf.cond(apply_aug,
                             lambda: tf.math.add(
                                 tf.nn.erosion2d(tf.expand_dims(self.image, 0),
                                                 kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME'),
                                 tf.constant(1.0))[0, :, :, :],
                             lambda: self.image)

    def random_dilation(self, srate=0.4, rrate=1.0, prob=0.5):
        kernel = tf.py_func(self.get_kernel, [srate, rrate], tf.float32)
        apply_aug = tf.math.less(tf.random.uniform([]), tf.constant(prob))
        self.image = tf.cond(apply_aug,
                             lambda: tf.math.add(
                                 tf.nn.dilation2d(tf.expand_dims(self.image, 0),
                                                  kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME'),
                                 tf.constant(-1.0))[0, :, :, :],
                             lambda: self.image)
