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

    def random_scaling(self, stdv=0.12, prob=0.5):
        if np.random.random() < prob:
            scale_factor = np.random.lognormal(mean=0, sigma=stdv)
            scale_factor = tf.constant(scale_factor)
            self.width = tf.cast(tf.multiply(self.width_float, scale_factor), tf.int32)
            self.image = self.transform(self.image, [[scale_factor, 0, 0],
                                                     [0, scale_factor, 0],
                                                     [0, 0, 1.0]],
                                        [self.height, self.width])

    def random_rotation(self, prec=100, prob=0.5):
        if np.random.random() < prob:
            rot_factor = np.random.vonmises(mu=0, kappa=prec)
            height_pad = tf.cast(tf.math.scalar_mul(self.width_float, tf.math.sin(tf.constant(rot_factor))), tf.int32)
            if rot_factor < 0:
                height_pad = tf.abs(height_pad)
            self.image = tf.image.pad_to_bounding_box(
                self.image, tf.cast(tf.math.divide(height_pad, tf.constant(2, tf.int32)), tf.int32), 0,
                tf.math.add(self.height, height_pad), self.width)
            self.image = tf.contrib.image.rotate(self.image, rot_factor)
            self.image = tf.image.resize_images(self.image, (self.height, self.width))

    def random_shearing(self, prec=4, prob=0.5):
        if np.random.random() < prob:
            shear_factor = np.random.vonmises(mu=0, kappa=prec)
            self.image = self.transform(self.image, [[1.0, shear_factor, 0],
                                                     [0, 1.0, 0],
                                                     [0, 0, 1.0]])

    def random_translation(self, stdv=0.02, prob=0.5):
        if np.random.random() < prob:
            trans_factor = np.random.normal(loc=0, scale=stdv, size=2)
            trans_factor_w = tf.constant(trans_factor[0], tf.float32)
            trans_factor_h = tf.constant(trans_factor[1], tf.float32)
            self.image = tf.contrib.image.translate(self.image, [trans_factor_w*self.width_float,
                                                                 trans_factor_h*self.height_float])

    def random_erosion(self, srate=0.8, rrate=1.2, prob=0.5):
        if np.random.random() < prob:
            allowed_kernel_sizes = [3, 5, 7, 9, 11, 15]
            kernel_size_probs = np.array([self.geom_prob(k, srate) for k in allowed_kernel_sizes])
            kernel_size_probs /= kernel_size_probs.sum()
            kernel_sizes = np.random.choice(allowed_kernel_sizes, 2, p=kernel_size_probs)
            kernel_shape = (kernel_sizes[0], kernel_sizes[1], 1)
            kernel = tf.convert_to_tensor(np.fromfunction(self.bernoulli_np_value, kernel_shape, dtype=np.float32,
                                                          array_shape=kernel_shape, r=rrate), dtype=tf.float32)
            self.image = tf.expand_dims(self.image, 0)
            self.image = tf.math.add(tf.nn.erosion2d(self.image, kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME'),
                                     tf.constant(1.0))
            self.image = self.image[0, :, :, :]

    def random_dilation(self, srate=0.4, rrate=1.0, prob=0.5):
        if np.random.random() < prob:
            allowed_kernel_sizes = [3, 5, 7, 9, 11, 15]
            kernel_size_probs = np.array([self.geom_prob(k, srate) for k in allowed_kernel_sizes])
            kernel_size_probs /= kernel_size_probs.sum()
            kernel_sizes = np.random.choice(allowed_kernel_sizes, 2, p=kernel_size_probs)
            kernel_shape = (kernel_sizes[0], kernel_sizes[1], 1)
            kernel = tf.convert_to_tensor(np.fromfunction(self.bernoulli_np_value, kernel_shape, dtype=np.float32,
                                                          array_shape=kernel_shape, r=rrate), dtype=tf.float32)
            self.image = tf.expand_dims(self.image, 0)
            self.image = tf.math.add(tf.nn.dilation2d(self.image, kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME'),
                                     tf.constant(-1.0))
            self.image = self.image[0, :, :, :]
