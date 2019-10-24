import sys

sys.path.extend(['..'])

import tensorflow as tf
from trainers.trainer import Trainer
from data_loader.data_generator import DataGenerator
from utils.config import process_config
from utils.utils import get_args
from importlib import import_module


def eval():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Get Model
    model_types = import_module('models.' + config.architecture + '_model')
    Model = getattr(model_types, 'Model')

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data_loader = DataGenerator(config, eval_phase=True, eval_on_test_data=True)

    # create instance of the model you want
    model = Model(data_loader, config)

    # create trainer and path all previous components to it
    trainer = Trainer(sess, model, config, None, data_loader, load_best=True)

    # here we evaluate on the test dataset
    test_loss, test_cer = trainer.test(tqdm_enable=True)
    print('\nTest set Loss:', test_loss)
    print('Test set CER:', round(test_cer*100, 2), '%')


if __name__ == '__main__':
    eval()
