from base.base_model import BaseModel
import tensorflow as tf
import warpctc_tensorflow


class BlstmModel(BaseModel):
    def __init__(self, data_loader, config):
        super(BlstmModel, self).__init__(config)
        self.rnn_num_hidden = 300
        self.rnn_num_layers = 3

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
    def calc_ler(predicted, targets):
        return tf.edit_distance(tf.cast(predicted, tf.int32), targets, normalize=True)

    def build_model(self):
        # Helper Variables
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        # Inputs to the network
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_input()
            self.y = tf.contrib.layers.dense_to_sparse(self.y, eos_token=-1)
            self.x, self.length, self.lab_length = self.x
            self.x = tf.transpose(self.x, [1, 0, 2])
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.length)
        tf.add_to_collection('inputs', self.lab_length)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        # Network Architecture
        out_W = tf.Variable(tf.truncated_normal([2 * self.rnn_num_hidden, self.data_loader.num_classes], stddev=0.1),
                            name='out_W')
        out_b = tf.Variable(tf.constant(0., shape=[self.data_loader.num_classes]), name='out_b')

        # RNN
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE):
            stacked_rnn = []
            for i in range(self.rnn_num_layers):
                stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_num_hidden, state_is_tuple=True))
            with tf.variable_scope('forward'):
                cell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
            with tf.variable_scope('backward'):
                cell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
            output, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                inputs=self.x,
                dtype=tf.float32,
                sequence_length=self.length,
                time_major=True,
                scope='MultiRNN'
            )

        # Fully Connected
        with tf.name_scope('Dense'):
            output = tf.concat(output, 2)
            # Reshaping to apply the same weights over the timesteps
            output = tf.reshape(output, [-1, 2*self.rnn_num_hidden])
            # Doing the affine projection
            logits = tf.matmul(output, out_W) + out_b

        # Reshaping back to the original shape
        self.logits = tf.reshape(logits, [self.config.batch_size, -1, self.data_loader.num_classes])
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        tf.add_to_collection('out', self.logits)

        with tf.variable_scope('loss-acc'):
            self.loss = tf.reduce_mean(
                warpctc_tensorflow.ctc(self.logits, self.y.values,
                                       self.lab_length, self.length, self.data_loader.num_classes - 1))
            self.prediction = tf.nn.ctc_beam_search_decoder(self.logits, sequence_length=self.length,
                                                            merge_repeated=True)
            self.ler = self.calc_ler(self.prediction[0][0], self.y)

        with tf.variable_scope('train_step'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.ler)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
