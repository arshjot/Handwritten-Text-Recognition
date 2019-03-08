from base.base_model import BaseModel
import tensorflow as tf
import warpctc_tensorflow


class Model(BaseModel):
    def __init__(self, data_loader, config):
        super(Model, self).__init__(config)
        self.rnn_num_hidden = 256
        self.rnn_num_layers = 4

        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x, self.y, self.length, self.lab_length = None, None, None, None
        self.is_training = None
        self.prediction = None
        self.loss = None
        self.ler = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    @staticmethod
    def calc_cer(predicted, targets):
        return tf.edit_distance(tf.cast(predicted, tf.int32), targets, normalize=True)

    def build_model(self):
        # Helper Variables
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        # Inputs to the network
        with tf.compat.v1.variable_scope('inputs'):
            self.x, y, self.length, self.lab_length = self.data_loader.get_input()
            self.y = tf.contrib.layers.dense_to_sparse(y, eos_token=-1)
            self.x = tf.transpose(a=self.x, perm=[2, 0, 1])
            self.is_training = tf.compat.v1.placeholder(tf.bool, name='Training_flag')
        tf.compat.v1.add_to_collection('inputs', self.x)
        tf.compat.v1.add_to_collection('inputs', self.length)
        tf.compat.v1.add_to_collection('inputs', self.lab_length)
        tf.compat.v1.add_to_collection('inputs', y)
        tf.compat.v1.add_to_collection('inputs', self.is_training)

        # Network Architecture
        out_W = tf.Variable(tf.random.truncated_normal([2 * self.rnn_num_hidden, self.data_loader.num_classes], stddev=0.1),
                            name='out_W')
        out_b = tf.Variable(tf.constant(0., shape=[self.data_loader.num_classes]), name='out_b')

        # RNN
        output = self.x
        with tf.compat.v1.variable_scope('MultiRNN', reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(self.rnn_num_layers):
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, self.rnn_num_hidden, 'linear_input', 'bidirectional')
                output, state = lstm(output)
                if i < self.rnn_num_layers - 1:
                    output = tf.compat.v1.layers.dropout(output, self.rnn_dropout, noise_shape=tf.constant(
                        value=[1, self.config.batch_size, 2 * self.rnn_num_hidden]), training=self.is_training)


        # Fully Connected
        with tf.compat.v1.name_scope('Dense'):
            output = tf.concat(output, 2)
            # Reshaping to apply the same weights over the timesteps
            output = tf.reshape(output, [-1, 2*self.rnn_num_hidden])
            # Doing the affine projection
            logits = tf.matmul(output, out_W) + out_b

        # Reshaping back to the original shape
        self.logits = tf.reshape(logits, [self.config.batch_size, -1, self.data_loader.num_classes])
        self.logits = tf.transpose(a=self.logits, perm=(1, 0, 2))

        with tf.compat.v1.variable_scope('loss-acc'):
            self.loss = warpctc_tensorflow.ctc(self.logits, self.y.values, self.lab_length, self.length,
                                               self.data_loader.num_classes - 1)
            self.cost = tf.reduce_mean(input_tensor=self.loss)
            self.prediction = tf.compat.v1.nn.ctc_beam_search_decoder(inputs=self.logits, sequence_length=self.length,
                                                            merge_repeated=True)
            self.cer = self.calc_cer(self.prediction[0][0], self.y)

        with tf.compat.v1.variable_scope('train_step'):
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)

        tf.compat.v1.add_to_collection('train', self.train_step)
        tf.compat.v1.add_to_collection('train', self.cost)
        tf.compat.v1.add_to_collection('train', self.cer)

    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
