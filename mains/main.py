import sys

sys.path.extend(['..'])

import tensorflow as tf
import shutil
import os
from data_loader.data_generator import DataGenerator
from models.BLSTM_model import BlstmModel
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    if config.train_from_start:
        if os.path.exists(config.summary_dir):
            shutil.rmtree(config.summary_dir)
        if os.path.exists(config.checkpoint_dir):
            shutil.rmtree(config.checkpoint_dir)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data_loader = DataGenerator(config)

    # create instance of the model you want
    model = BlstmModel(data_loader, config)

    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                               scalar_tags=['train/loss_per_epoch', 'train/ler_per_epoch',
                                            'test/loss_per_epoch', 'test/ler_per_epoch'])

    # create trainer and path all previous components to it
    trainer = Trainer(sess, model, config, logger, data_loader)

    # here we train our model
    trainer.train()


if __name__ == '__main__':
    main()
