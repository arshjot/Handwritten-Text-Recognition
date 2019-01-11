from base.base_model import BaseModel
import tensorflow as tf
import warpctc_tensorflow


class BlstmModel(BaseModel):
    def __init__(self, config):
        super(BlstmModel, self).__init__(config)
        self.build_model()
        self.init_saver()
        self.rnn_num_hidden = 200
        self.rnn_num_layers = 3

    @staticmethod
    def calc_ler(predicted, targets):
        return tf.edit_distance(tf.cast(predicted, tf.int32), targets, normalize=True)

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, self.config.im_height, self.config.im_width, 1], name='image')
        self.y = tf.sparse_placeholder(tf.int64, name='label')
        self.length = tf.placeholder(tf.int16, shape=[None], name='image width')
        self.lab_length = tf.placeholder(tf.int16, shape=[None], name='label length')

        # Network Architecture
        out_W = tf.Variable(tf.truncated_normal([2 * self.rnn_num_hidden, self.config.num_classes], stddev=0.1),
                            name='out_W')
        out_b = tf.Variable(tf.constant(0., shape=[self.config.num_classes]), name='out_b')

        # RNN
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE):
            stacked_rnn = []
            for i in range(self.rnn_num_layers):
                stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_num_hidden, state_is_tuple=True))
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
                scope='MultiRNN'
            )

        # Fully Connected
        with tf.name_scope('Dense'):
            output = tf.concat(output, 2)
            # Reshaping to apply the same weights over the timesteps
            output = tf.reshape(output, [-1, 2 * self.rnn_num_hidden])
            # Doing the affine projection
            logits = tf.matmul(output, out_W) + out_b

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [self.config.batch_size, -1, self.config.num_classes])

        with tf.name_scope("loss"):
            self.ctc_loss = tf.reduce_mean(
                warpctc_tensorflow.ctc(logits, self.y.values, self.lab_length, self.length,
                                       self.config.num_classes-1))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=0.95).minimize(
                    self.ctc_loss, global_step=self.global_step_tensor)
            prediction = tf.nn.ctc_beam_search_decoder(logits, sequence_length=self.length, merge_repeated=True)

            # Calculate ler
            self.ler = self.calc_ler(prediction[0][0], self.y)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

