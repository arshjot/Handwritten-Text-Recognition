import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from data_loader.data_generator import DataGenerator
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
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

    # create the experiments dirs
    if config.train_from_start:
        if os.path.exists(config.summary_dir):
            shutil.rmtree(config.summary_dir)
        if os.path.exists(config.checkpoint_dir):
            shutil.rmtree(config.checkpoint_dir)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # Get Model
    model_types = import_module('models.' + config.architecture + '_model')
    Model = getattr(model_types, 'Model')

    # create your data generator
    data_loader = DataGenerator(config)

    # create instance of the model you want
    model = Model(data_loader, config)

    # create trainer and path all previous components to it
    trainer = Trainer(model, data_loader, config)

    # here we train our model
    trainer.train()


if __name__ == '__main__':
    main()
