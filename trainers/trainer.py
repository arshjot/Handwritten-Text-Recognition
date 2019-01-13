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

        self.x,self.length, self.lab_length,  self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node = tf.get_collection('train')

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
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
        losses = []
        lers = []
        for _ in tt:
            loss, ler = self.train_step()
            losses.append(loss)
            lers.append(ler)
            progbar.update(_, values=[('loss', loss), ('ler', ler)])
        loss = np.mean(losses)
        ler = np.mean(lers)

        self.sess.run(self.model.global_epoch_inc)

        # summarize
        summaries_dict = {
            'train/loss_per_epoch': loss,
            'train/ler_per_epoch': ler,
        }
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)

        print("""Epoch-{}  loss:{:.4f} -- ler:{:.4f}""".format(epoch, loss, ler))

    def train_step(self):
        """
        Run the session of train_step in tensorflow, also get the loss & acc of that minibatch.
        :return: (loss, ler) tuple of some metrics to be used in summaries
        """
        _, loss, ler = self.sess.run([self.train_op, self.loss_node, self.acc_node],
                                     feed_dict={self.is_training: True})
        return loss, ler

    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=False)

        # initialize tqdm
        tt = range(self.data_loader.num_iterations_val)

        losses = []
        lers = []
        # Iterate over batches
        for _ in tt:
            loss, ler = self.sess.run([self.loss_node, self.acc_node],
                                      feed_dict={self.is_training: False})
            losses.append(loss)
            lers.append(ler)
        loss = np.mean(losses)
        ler = np.mean(lers)

        # summarize
        summaries_dict = {
            'test/loss_per_epoch': loss,
            'test/ler_per_epoch': ler,
        }
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""Val-{}  loss:{:.4f} -- ler:{:.4f}""".format(epoch, loss, ler))
