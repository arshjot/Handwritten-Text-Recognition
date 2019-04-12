import argparse
import numpy as np
import tensorflow as tf


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representation of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


class RandomPrediction(tf.keras.callbacks.Callback):
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        for sample in self.data_loader.val_dataset:
            predictions = self.model.predict(sample)
            label = self.model.y
            break
        pred = np.zeros(predictions[0][0].dense_shape)
        for idx, val in enumerate(predictions[0][0].indices):
            pred[val[0]][val[1]] = predictions[0][0].values[idx]
        pred = ''.join([self.data_loader.char_map_inv[i] for i in pred[0]])
        end_idx = np.where(label[0] == -1)[0][0] if len(np.where(label[0] == -1)[0]) != 0 else len(label[0])
        label = ''.join([self.data_loader.char_map_inv[i] for i in label[0][:end_idx]])
        print("Label: {}\nPrediction: {}".format(label, pred))
