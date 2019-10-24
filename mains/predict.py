import sys

sys.path.extend(['..'])

import tensorflow as tf
import glob
from data_loader.data_generator import DataGenerator
from utils.config import process_config
from utils.utils import get_args
from importlib import import_module
from tqdm import tqdm
import numpy as np
from data.process_images import process_ims
import shutil


def predict():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Pre-process images
    print('Pre-process images:')
    process_ims(input_dir='../samples/*', output_dir='../samples/processed/', out_height=config.im_height)

    # Get Model
    model_types = import_module('models.' + config.architecture + '_model')
    Model = getattr(model_types, 'Model')

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data_loader = DataGenerator(config, eval_phase=True, eval_on_test_data=False)

    # create instance of the model you want
    model = Model(data_loader, config)

    # load the model from the best checkpoint
    model.load(sess, config.best_model_dir)

    x, length, lab_length, y, is_training = tf.get_collection('inputs')
    pred = model.prediction

    # initialize dataset
    data_loader.initialize(sess, is_train=False)

    # Progress bar
    tt = range(data_loader.num_iterations_val)

    # Iterate over batches
    predictions, filenames = [], sorted(glob.glob('../samples/*'))
    for _ in tqdm(tt):
        preds_sparse = sess.run([pred], feed_dict={is_training: False})

        # Map numeric predictions with corresponding character labels
        preds_out = np.zeros(preds_sparse[0][0][0].dense_shape)
        for idx, val in enumerate(preds_sparse[0][0][0].indices):
            preds_out[val[0]][val[1]] = preds_sparse[0][0][0].values[idx]
        predictions += [''.join([data_loader.char_map_inv[j] for j in preds_out[i]]) for i in range(preds_out.shape[0])]

    print('\nPredictions:')
    [print('{}: {}'.format(name[name.rfind('/')+1:], model_pred)) for name, model_pred in zip(filenames, predictions)]
    shutil.rmtree('../samples/processed/')


if __name__ == '__main__':
    predict()
