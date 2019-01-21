from base.base_model import BaseModel
import tensorflow as tf
import warpctc_tensorflow


class Model(BaseModel):
    def __init__(self, data_loader, config):
        super(Model, self).__init__(config)
        self.rnn_num_hidden = 256
        self.rnn_num_layers = 4
        self.conv1_patch_size = 3
        self.conv2_patch_size = 3
        self.conv3_patch_size = 3
        self.conv4_patch_size = 3
        self.conv1_depth = 12
        self.conv2_depth = 24
        self.conv3_depth = 48
        self.conv4_depth = 96
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
            self.x, y = self.data_loader.get_input()
            self.y = tf.contrib.layers.dense_to_sparse(y, eos_token=-1)
            self.x, self.length, self.lab_length = self.x
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
        # ==================1==================
        conv1_out = tf.layers.conv2d(self.x, self.conv1_depth, self.conv1_patch_size, padding='same',
                                     activation=tf.nn.leaky_relu, kernel_initializer=intitalizer, name='cnn_1')
        conv1_out = tf.layers.max_pooling2d(conv1_out, 2, 2, padding='same', name='pool_1')

        # ==================2==================
        conv2_out = tf.layers.conv2d(conv1_out, self.conv2_depth, self.conv2_patch_size, padding='same',
                                     activation=tf.nn.leaky_relu, kernel_initializer=intitalizer, name='cnn_2')
        conv2_out = tf.layers.max_pooling2d(conv2_out, 2, 2, padding='same', name='pool_2')

        # ==================3==================
        conv3_out = tf.layers.conv2d(conv2_out, self.conv3_depth, self.conv3_patch_size, padding='same',
                                     activation=tf.nn.leaky_relu, kernel_initializer=intitalizer, name='cnn_3')
        conv3_out = tf.layers.max_pooling2d(conv3_out, 2, 2, padding='same', name='pool_3')

        # ==================4==================
        conv4_out = tf.layers.conv2d(conv3_out, self.conv4_depth, self.conv4_patch_size, padding='same',
                                     activation=tf.nn.leaky_relu, kernel_initializer=intitalizer, name='cnn_4')

        cnn_out = tf.reshape(conv4_out, [self.config.batch_size, -1,
                                         self.conv4_depth*self.config.im_height//self.reduce_factor])
        cnn_out = tf.transpose(cnn_out, [1, 0, 2])
        cnn_out = tf.layers.dropout(cnn_out, 0.8, training=self.is_training, name='drop_cnn')

        # RNN
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE) as sc:
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(self.rnn_num_layers, self.rnn_num_hidden,
                                                  'linear_input', 'bidirectional', name=sc)
            output, state = lstm(cnn_out)

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
                self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.cost)
        tf.add_to_collection('train', self.cer)
        tf.add_to_collection('sample_pred', self.prediction[0][0].values)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
