from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Conv2D, LeakyReLU, BatchNormalization, MaxPool2D, Dense, LSTM, \
    Bidirectional, Masking, Lambda, Multiply, Reshape
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

    @staticmethod
    def calc_cer(predicted, targets):
        return tf.edit_distance(tf.cast(predicted, tf.int32), targets, normalize=True)

    def build_model(self, eval_mode=False):
        # Inputs to the network
        self.x = tf.keras.Input(shape=(self.config.im_height, None,))
        self.length = tf.keras.Input(shape=(), dtype=tf.int32)
        self.lab_length = tf.keras.Input(shape=(1,), dtype=tf.int32)
        self.y = tf.keras.Input(shape=(None,), sparse=True, dtype=tf.int32)

        self.x_in = Reshape((self.config.im_height, -1, 1))(self.x)
        self.length_in = self.length
        batch_size = tf.shape(self.x_in)[0]

        intitalizer = 'glorot_uniform'
        # CNN Block 1
        conv1_out = Dropout(self.conv_dropouts[0])(self.x_in)
        conv1_out = Conv2D(self.conv_depths[0], self.conv_patch_sizes[0],
                           padding='same', activation=None, kernel_initializer=intitalizer)(conv1_out)
        conv1_out = BatchNormalization()(conv1_out)
        conv1_out = LeakyReLU(alpha=0.01)(conv1_out)
        conv1_out = MaxPool2D(2, 2, padding='same')(conv1_out)

        # CNN Block 2
        conv2_out = Dropout(self.conv_dropouts[1])(conv1_out)
        conv2_out = Conv2D(self.conv_depths[1], self.conv_patch_sizes[1],
                           padding='same', activation=None, kernel_initializer=intitalizer)(conv2_out)
        conv2_out = BatchNormalization()(conv2_out)
        conv2_out = LeakyReLU(alpha=0.01)(conv2_out)
        conv2_out = MaxPool2D(2, 2, padding='same')(conv2_out)

        # CNN Block 3
        conv3_out = Dropout(self.conv_dropouts[2])(conv2_out)
        conv3_out = Conv2D(self.conv_depths[2], self.conv_patch_sizes[2],
                           padding='same', activation=None, kernel_initializer=intitalizer)(conv3_out)
        conv3_out = BatchNormalization()(conv3_out)
        conv3_out = LeakyReLU(alpha=0.01)(conv3_out)
        conv3_out = MaxPool2D(2, 2, padding='same')(conv3_out)

        # CNN Block 4
        conv4_out = Dropout(self.conv_dropouts[3])(conv3_out)
        conv4_out = Conv2D(self.conv_depths[3], self.conv_patch_sizes[3],
                           padding='same', activation=None, kernel_initializer=intitalizer)(conv4_out)
        conv4_out = BatchNormalization()(conv4_out)
        conv4_out = LeakyReLU(alpha=0.01)(conv4_out)

        # CNN Block 5
        conv5_out = Dropout(self.conv_dropouts[4])(conv4_out)
        conv5_out = Conv2D(self.conv_depths[4], self.conv_patch_sizes[4],
                           padding='same', activation=None, kernel_initializer=intitalizer)(conv5_out)
        conv5_out = BatchNormalization()(conv5_out)
        conv5_out = LeakyReLU(alpha=0.01)(conv5_out)

        output = Lambda(lambda x: tf.transpose(a=x, perm=[2, 0, 1, 3]))(conv5_out)
        output = Lambda(lambda x: tf.reshape(
            x, [-1, batch_size, (self.config.im_height // self.reduce_factor)*self.conv_depths[4]]))(output)

        # RNN
        # Make values after length as 0 and implement it using keras Masking layer
        mask = Lambda(lambda x: tf.transpose(tf.sequence_mask(x, maxlen=tf.shape(output)[0],
                                                              dtype=tf.float32)))(self.length_in)
        output = Multiply()([output, mask])
        output = Masking()(output)

        for _ in range(self.rnn_num_layers):
            output = Bidirectional(LSTM(self.rnn_num_hidden, dropout=self.rnn_dropout,
                                        time_major=True, return_sequences=True))(output)

        # Fully Connected
        output = Dropout(self.linear_dropout)(output)
        # Reshaping to apply the same weights over the timesteps
        output = Lambda(lambda x: tf.reshape(x, [-1, 2*self.rnn_num_hidden]))(output)
        # Doing the affine projection
        logits = Dense(self.data_loader.num_classes)(output)

        # Reshaping back to the original shape
        self.logits = Lambda(lambda x: tf.reshape(x, [-1, batch_size, self.data_loader.num_classes]))(logits)

        # self.prediction = tf.nn.ctc_beam_search_decoder(inputs=self.logits, sequence_length=self.length_in)
        # self.cer = self.calc_cer(self.prediction[0][0], self.y)

        if eval_mode:
            self.model = tf.keras.Model(inputs=[self.x, self.length], output=self.logits)
        else:
            self.loss = warpctc_tensorflow.ctc(self.logits, self.y.values, self.lab_length, self.length_in,
                                               self.data_loader.num_classes - 1)
            self.cost = tf.reduce_mean(input_tensor=self.loss)
            self.model = tf.keras.Model([self.x, self.length, self.lab_length, self.y], self.logits)
            # self.model.add_loss(self.loss)
        # self.model.add_metric(self.cer, name='cer', aggregation='mean')
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate,
                                               rho=self.config.learning_rate_decay)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=None)
