from base.base_train import BaseTrain
import numpy as np
import tensorflow as tf


class Trainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        super(Trainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        self.model.load(self.sess)

        # Summarizer
        self.summarizer = logger

        self.x, self.length, self.lab_length,  self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        """
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=True)

        # initialize tqdm
        tt = range(self.data_loader.num_iterations_train)
        progbar = tf.keras.utils.Progbar(self.data_loader.num_iterations_train)

        # Iterate over batches
        losses, cers = [], []

        for _ in tt:
            # Beam search prediction at every 10th step only - to improve training speed
            if _ % 10 == 0:
                loss, cer = self.train_step()
                cers.append(cer)
                losses.append(loss)
                progbar.update(_, values=[('loss', loss), ('cer', cer)])
            else:
                loss = self.train_step(get_err=False)
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

        self.model.save(self.sess)

        print("""\tEpoch-{}: Train - loss:{:.4f} -- cer:{:.4f}""".format(epoch+1, loss, cer), end="")

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

    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=False)

        # initialize tqdm
        tt = range(self.data_loader.num_iterations_val)

        losses, cers = [], []
        # Iterate over batches
        for _ in tt:
            loss, cer = self.sess.run([self.loss_node, self.acc_node],
                                      feed_dict={self.is_training: False})
            losses.append(loss)
            cers.append(cer)
        loss = np.mean(losses)
        cer = np.mean(cers)

        # summarize
        summaries_dict = {
            'test/loss_per_epoch': loss,
            'test/cer_per_epoch': cer,
        }
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""\tVal - loss:{:.4f} -- cer:{:.4f}""".format(loss, cer))
