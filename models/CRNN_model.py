from base.base_model import BaseModel
import tensorflow as tf
import warpctc_tensorflow


class Model(BaseModel):
    def __init__(self, data_loader, config):
        super(Model, self).__init__(config)
        self.rnn_num_hidden = 256
        self.rnn_num_layers = 5
        self.rnn_dropout = 0.5
        self.conv_patch_sizes = [3] * 5
        self.conv_depths = [16, 32, 48, 64, 80]
        self.conv_dropouts = [0, 0, 0.2, 0.2, 0.2]
        self.linear_dropout = 0.5
        self.reduce_factor = 8

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
        with tf.variable_scope('inputs'):
            self.x, y, self.length, self.lab_length = self.data_loader.get_input()
            self.y = tf.contrib.layers.dense_to_sparse(y, eos_token=-1)
            self.length = tf.div(self.length, tf.constant(self.reduce_factor, dtype=tf.int32))
            self.x = tf.expand_dims(self.x, 3)
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.length)
        tf.add_to_collection('inputs', self.lab_length)
        tf.add_to_collection('inputs', y)
        tf.add_to_collection('inputs', self.is_training)

        # Define CNN variables
        intitalizer = tf.contrib.layers.xavier_initializer_conv2d()

        out_W = tf.Variable(tf.truncated_normal([2 * self.rnn_num_hidden, self.data_loader.num_classes], stddev=0.1),
                            name='out_W')
        out_b = tf.Variable(tf.constant(0., shape=[self.data_loader.num_classes]), name='out_b')

        # CNNs
        with tf.name_scope('CNN_Block_1'):
            conv1_out = tf.layers.conv2d(self.x, self.conv_depths[0], self.conv_patch_sizes[0], padding='same',
                                         activation=tf.nn.leaky_relu, kernel_initializer=intitalizer)
            conv1_out = tf.layers.max_pooling2d(conv1_out, 2, 2, padding='same')
            conv1_out = tf.layers.dropout(conv1_out, self.conv_dropouts[0], noise_shape=tf.constant(
                value=[self.config.batch_size, 1, 1, self.conv_depths[0]]), training=self.is_training)
            conv1_out = tf.layers.batch_normalization(conv1_out)

        with tf.name_scope('CNN_Block_2'):
            conv2_out = tf.layers.conv2d(conv1_out, self.conv_depths[1], self.conv_patch_sizes[1], padding='same',
                                         activation=tf.nn.leaky_relu, kernel_initializer=intitalizer)
            conv2_out = tf.layers.max_pooling2d(conv2_out, 2, 2, padding='same')
            conv2_out = tf.layers.dropout(conv2_out, self.conv_dropouts[1], noise_shape=tf.constant(
                value=[self.config.batch_size, 1, 1, self.conv_depths[1]]), training=self.is_training)
            conv2_out = tf.layers.batch_normalization(conv2_out)

        with tf.name_scope('CNN_Block_3'):
            conv3_out = tf.layers.conv2d(conv2_out, self.conv_depths[2], self.conv_patch_sizes[2], padding='same',
                                         activation=tf.nn.leaky_relu, kernel_initializer=intitalizer)
            conv3_out = tf.layers.max_pooling2d(conv3_out, 2, 2, padding='same')
            conv3_out = tf.layers.dropout(conv3_out, self.conv_dropouts[2], noise_shape=tf.constant(
                value=[self.config.batch_size, 1, 1, self.conv_depths[2]]), training=self.is_training)
            conv3_out = tf.layers.batch_normalization(conv3_out)

        with tf.name_scope('CNN_Block_4'):
            conv4_out = tf.layers.conv2d(conv3_out, self.conv_depths[3], self.conv_patch_sizes[3], padding='same',
                                         activation=tf.nn.leaky_relu, kernel_initializer=intitalizer)
            conv4_out = tf.layers.dropout(conv4_out, self.conv_dropouts[3], noise_shape=tf.constant(
                value=[self.config.batch_size, 1, 1, self.conv_depths[3]]), training=self.is_training)
            conv4_out = tf.layers.batch_normalization(conv4_out)

        with tf.name_scope('CNN_Block_5'):
            conv5_out = tf.layers.conv2d(conv4_out, self.conv_depths[4], self.conv_patch_sizes[4], padding='same',
                                         activation=tf.nn.leaky_relu, kernel_initializer=intitalizer)
            conv5_out = tf.layers.dropout(conv5_out, self.conv_dropouts[4], noise_shape=tf.constant(
                value=[self.config.batch_size, 1, 1, self.conv_depths[4]]), training=self.is_training)
            conv5_out = tf.layers.batch_normalization(conv5_out)

        cnn_out = tf.reduce_sum(conv5_out, axis=1)
        output = tf.transpose(cnn_out, [1, 0, 2])

        # RNN
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE):
            for i in range(self.rnn_num_layers):
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, self.rnn_num_hidden, 'linear_input', 'bidirectional')
                output, state = lstm(output)
                if i < self.rnn_num_layers - 1:
                    output = tf.layers.dropout(output, self.rnn_dropout, training=self.is_training)

        # Fully Connected
        with tf.name_scope('Dense'):
            output = tf.concat(output, 2)
            # Linear dropout
            output = tf.layers.dropout(output, self.linear_dropout, training=self.is_training)
            # Reshaping to apply the same weights over the timesteps
            output = tf.reshape(output, [-1, 2*self.rnn_num_hidden])
            # Doing the affine projection
            logits = tf.matmul(output, out_W) + out_b

        # Reshaping back to the original shape
        self.logits = tf.reshape(logits, [self.config.batch_size, -1, self.data_loader.num_classes])
        self.logits = tf.transpose(self.logits, (1, 0, 2))

        with tf.variable_scope('loss-acc'):
            self.loss = warpctc_tensorflow.ctc(self.logits, self.y.values, self.lab_length, self.length,
                                               self.data_loader.num_classes - 1)
            self.cost = tf.reduce_mean(self.loss)
            self.prediction = tf.nn.ctc_beam_search_decoder(self.logits, sequence_length=self.length,
                                                            merge_repeated=True)
            self.cer = self.calc_cer(self.prediction[0][0], self.y)

        with tf.variable_scope('train_step'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate,
                                                            decay=self.config.learning_rate_decay).minimize(
                    self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.cost)
        tf.add_to_collection('train', self.cer)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
