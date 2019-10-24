from base.base_train import BaseTrain
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Trainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader, load_best=False):
        super(Trainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        if load_best:
            self.model.load(self.sess, self.config.best_model_dir)
        else:
            self.model.load(self.sess, self.config.checkpoint_dir)

        # Summarizer
        self.summarizer = logger

        self.x, self.length, self.lab_length,  self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')
        self.pred = self.model.prediction

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        """
        best_val = 10
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.model.save(self.sess)
            curr_val_loss, curr_val_cer = self.test()
            if cur_epoch == 1:
                best_val = curr_val_cer
            if curr_val_cer < best_val:
                self.model.save(self.sess, True)
                best_val = curr_val_cer

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        """
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=True)

        # Progress bar
        tt = range(self.data_loader.num_iterations_train)
        progbar = tf.keras.utils.Progbar(self.data_loader.num_iterations_train)

        # Iterate over batches
        losses, cers = [], []

        for _ in tt:
            # Beam search prediction at every 10th step only - to improve training speed
            if _ % 10 == 0:
                loss, cer = self.train_step()
                cers += list(cer)
                losses.append(loss)
                progbar.update(_, values=[('loss', loss), ('cer', np.mean(cer))])
            else:
                loss = self.train_step(get_err=False)
                losses.append(loss)
                progbar.update(_, values=[('loss', loss)])

        loss = np.mean(losses)
        cer = np.mean(cers)

        self.sess.run(self.model.global_epoch_inc)

        # summarize
        summaries_dict = {
            'train/loss_per_epoch': loss,
            'train/cer_per_epoch': cer,
        }
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""\tEpoch-{}: Train - loss:{:.4f} -- cer:{:.4f}""".format(epoch, loss, cer), end="")

    def train_step(self, get_err=True):
        """
        Run the session of train_step in tensorflow, also get the loss & acc of that minibatch.
        :return: (loss, ler) tuple of some metrics to be used in summaries
        """
        if get_err:
            _, loss, cer = self.sess.run([self.train_op, self.loss_node, self.acc_node],
                                         feed_dict={self.is_training: True})
            return loss, cer
        else:
            _, loss = self.sess.run([self.train_op, self.loss_node],
                                    feed_dict={self.is_training: True})
            return loss

    def test(self, tqdm_enable=False):
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=False)

        # Progress bar
        tt = range(self.data_loader.num_iterations_val)
        sample_num = np.random.choice(tt)

        losses, cers = [], []
        # Iterate over batches
        for _ in tqdm(tt, disable=not tqdm_enable):
            if self.config.random_prediction & (_ == sample_num):
                loss, cer, predictions, label = self.sess.run([self.loss_node, self.acc_node, self.pred, self.y],
                                                              feed_dict={self.is_training: False})
            else:
                loss, cer = self.sess.run([self.loss_node, self.acc_node],
                                          feed_dict={self.is_training: False})
            losses.append(loss)
            cers += list(cer)
        loss = np.mean(losses)
        cer = np.mean(cers)

        # summarize
        summaries_dict = {
            'test/loss_per_epoch': loss,
            'test/cer_per_epoch': cer,
        }
        if self.summarizer is not None:
            self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""\tVal - loss:{:.4f} -- cer:{:.4f}""".format(loss, cer))

        if self.config.random_prediction:  # Print random prediction and corresponding label
            pred = np.zeros(predictions[0][0].dense_shape)
            for idx, val in enumerate(predictions[0][0].indices):
                pred[val[0]][val[1]] = predictions[0][0].values[idx]
            pred = ''.join([self.data_loader.char_map_inv[i] for i in pred[0]])
            end_idx = np.where(label[0] == -1)[0][0] if len(np.where(label[0] == -1)[0]) != 0 else len(label[0])
            label = ''.join([self.data_loader.char_map_inv[i] for i in label[0][:end_idx]])
            print("Label: {}\nPrediction: {}".format(label, pred))

        return loss, cer
