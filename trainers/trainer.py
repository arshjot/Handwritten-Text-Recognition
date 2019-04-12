from base.base_train import BaseTrain
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from utils.utils import RandomPrediction


class Trainer(BaseTrain):
    def __init__(self, model, data_loader, config):
        super(Trainer, self).__init__(model, data_loader, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

        self.data_loader = data_loader

        self.init_callbacks()
        self.pred = self.model.prediction

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir,
                                      '%s-{epoch:02d}-{val_cer:.2f}.hdf5' % self.config.exp_name),
                monitor='val_cer',
                save_best_only=True,
                save_weights_only=True,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.summary_dir,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor='val_acc',
                patience=80,
                verbose=1,
                min_delta=1e-4,
            )
        )

        self.callbacks.append(
            ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.3,
                patience=20,
                verbose=1,
                min_delta=1e-4,
            )
        )

        # Show random prediction at end of epoch
        self.callbacks.append(
            RandomPrediction(
                data_loader=self.data_loader,
                model=self.model,
            )
        )

    @tf.function
    def train(self):
        history = self.model.model.fit(
            self.data_loader.train_dataset,
            epochs=self.config.num_epochs,
            validation_data=self.data_loader.val_dataset,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
