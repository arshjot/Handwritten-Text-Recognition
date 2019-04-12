import sys

sys.path.extend(['..'])

import tensorflow as tf

import shutil
import os
from data_loader.data_generator import DataGenerator
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args
from importlib import import_module


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Get Model
    model_types = import_module('models.' + config.architecture + '_model')
    Model = getattr(model_types, 'Model')

    # create the experiments dirs
    if config.train_from_start:
        if os.path.exists(config.summary_dir):
            shutil.rmtree(config.summary_dir)
        if os.path.exists(config.checkpoint_dir):
            shutil.rmtree(config.checkpoint_dir)
        if os.path.exists(config.best_model_dir):
            shutil.rmtree(config.best_model_dir)
    create_dirs([config.summary_dir, config.checkpoint_dir, config.best_model_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data_loader = DataGenerator(config)

    # create instance of the model you want
    model = Model(data_loader, config)

    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                               scalar_tags=['train/loss_per_epoch', 'train/cer_per_epoch',
                                            'test/loss_per_epoch', 'test/cer_per_epoch'])

    # create trainer and path all previous components to it
    trainer = Trainer(sess, model, config, logger, data_loader)

    # here we train our model
    trainer.train()


if __name__ == '__main__':
    main()
