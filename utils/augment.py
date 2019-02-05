import tensorflow as tf
import numpy as np
import math


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

    def archived_random_scaling(self, stdv=0.12, prob=0.5):
        if np.random.random() < prob:
            scale_factor = np.random.lognormal(mean=0, sigma=stdv)
            if scale_factor > 1:
                self.image = tf.image.central_crop(self.image, central_fraction=1/scale_factor)
                self.image = tf.image.resize_images(self.image, (self.height, self.width))
            else:
                scale_factor = tf.constant(scale_factor)
                target_height = tf.cast(tf.multiply(self.height_float, scale_factor), tf.int32)
                target_width = tf.cast(tf.multiply(self.width_float, scale_factor), tf.int32)
                self.image = tf.image.resize_images(self.image, (target_height, target_width))
                self.image = tf.image.resize_image_with_pad(self.image, self.height, self.width)
            return self.image

    def random_scaling(self, stdv=0.12, prob=0.5):
        if np.random.random() < prob:
            scale_factor = np.random.lognormal(mean=0, sigma=stdv)
            scale_factor = tf.constant(scale_factor)
            self.width = tf.cast(tf.multiply(self.width_float, scale_factor), tf.int32)
            return self.transform(self.image, [[scale_factor, 0, 0],
                                               [0, scale_factor, 0],
                                               [0, 0, 1.0]],
                                  [self.height, self.width])

    def random_rotation(self, prec=300, prob=0.5):
        if np.random.random() < prob:
            rot_factor = np.random.vonmises(mu=0, kappa=prec)
            height_pad = tf.cast(tf.math.scalar_mul(self.width_float, tf.math.sin(tf.constant(rot_factor))), tf.int32)
            if rot_factor < 0:
                height_pad = tf.abs(height_pad)
            self.image = tf.image.pad_to_bounding_box(
                self.image, tf.cast(tf.math.divide(height_pad, tf.constant(2, tf.int32)), tf.int32), 0,
                tf.math.add(self.height, height_pad), self.width)
            self.image = tf.contrib.image.rotate(self.image, rot_factor)
            return tf.image.resize_images(self.image, (self.height, self.width))

    def random_shearing(self, prec=4, prob=0.5):
        if np.random.random() < prob:
            shear_factor = np.random.vonmises(mu=0, kappa=prec)
            return self.transform(self.image, [[1.0, shear_factor, 0],
                                               [0, 1.0, 0],
                                               [0, 0, 1.0]])

    def random_translation(self, stdv=0.02, prob=0.5):
        if np.random.random() < prob:
            trans_factor = np.random.normal(loc=0, scale=stdv, size=2)
            trans_factor_w = tf.constant(trans_factor[0], tf.float32)
            trans_factor_h = tf.constant(trans_factor[1], tf.float32)
            return tf.contrib.image.translate(self.image, [trans_factor_w*self.width_float,
                                                           trans_factor_h*self.height_float])
