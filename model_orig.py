import tensorflow as tf
import numpy as np
import math

# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)


def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):
    print("-" * 20)
    print("Convolutional Block", str(num_filters), name)
    print("-" * 20)
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [
                    1,
                    inputs.get_shape()[2],
                    inputs.get_shape()[3], num_filters
                ]
                W = tf.get_variable(
                    name='W',
                    shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv2d(
                    inputs, W, strides=[1, 1, 1, 1], padding="SAME")
                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    momentum=0.997,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Conv1D:", inputs.get_shape())
    print("-" * 20)
    if shortcut is not None:
        print("-" * 5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-" * 5)
        return inputs + shortcut
    return inputs


# Three types of downsampling methods described by paper
def downsampling(inputs,
                 downsampling_type,
                 name,
                 optional_shortcut=False,
                 shortcut=None):
    # k-maxpooling
    if downsampling_type == 'k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(
            tf.transpose(inputs, [0, 2, 1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0, 2, 1])
    # Linear
    elif downsampling_type == 'linear':
        pool = tf.layers.conv1d(
            inputs=inputs,
            filters=inputs.get_shape()[2],
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False)
    # Maxpooling
    else:
        pool = tf.nn.max_pool(
            inputs,
            ksize=[1, 1, 3, 1],
            strides=[1, 1, 2, 1],
            padding='SAME',
            name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(
            inputs=shortcut,
            filters=shortcut.get_shape()[2],
            kernel_size=1,
            strides=2,
            padding='same',
            use_bias=False)
        print("-" * 5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-" * 5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv2d(
        inputs=pool,
        filters=pool.get_shape()[3] * 2,
        kernel_size=1,
        strides=1,
        padding='VALID',
        use_bias=False)


def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs,
                           [[0, 0], [0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


#  sequence_max_length=1024,
#  embedding_size=16,
#  num_quantized_chars=69,
class VDCNN():
    def __init__(self,
                 input_dim,
                 batchsize,
                 depth=9,
                 downsampling_type='maxpool',
                 use_he_uniform=True,
                 optional_shortcut=False):

        # Depth to No. Layers
        if depth == 9:
            num_layers = [2, 2, 2, 2]
        elif depth == 17:
            num_layers = [4, 4, 4, 4]
        elif depth == 29:
            num_layers = [10, 10, 4, 4]
        elif depth == 49:
            num_layers = [16, 16, 10, 6]
        else:
            raise ValueError('depth=%g is a not a valid setting!' % depth)

        # input tensors
        self.input_x = tf.placeholder(
            tf.float32, [batchsize, input_dim[0], input_dim[1], 1],
            name="noise_feat")
        self.input_y = tf.placeholder(
            tf.float32, [batchsize, input_dim[0], input_dim[1], 1],
            name="clean_feat")
        self.is_training = tf.placeholder(tf.bool)
        self.layers = []

        # First Conv Layer
        with tf.variable_scope("First_Conv"):
            filter_shape = [1, 3, 1, 64]
            W = tf.get_variable(
                name='W_1',
                shape=filter_shape,
                initializer=he_normal,
                regularizer=regularizer)
            inputs = tf.nn.conv2d(
                self.input_x, W, strides=[1, 1, 1, 1], padding="SAME")
            #inputs = tf.nn.relu(inputs)
        print("First Conv", inputs.get_shape())
        self.layers.append(inputs)

        # Conv Block 64
        for i in range(num_layers[0]):
            if i < num_layers[0] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(
                inputs=self.layers[-1],
                shortcut=shortcut,
                num_filters=64,
                is_training=self.is_training,
                name=str(i + 1))
            self.layers.append(conv_block)
        pool1 = downsampling(
            self.layers[-1],
            downsampling_type=downsampling_type,
            name='pool1',
            optional_shortcut=optional_shortcut,
            shortcut=self.layers[-2])
        self.layers.append(pool1)
        print("Pooling:", pool1.get_shape())

        # Conv Block 128
        for i in range(num_layers[1]):
            if i < num_layers[1] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(
                inputs=self.layers[-1],
                shortcut=shortcut,
                num_filters=128,
                is_training=self.is_training,
                name=str(i + 1))
            self.layers.append(conv_block)
        pool2 = downsampling(
            self.layers[-1],
            downsampling_type=downsampling_type,
            name='pool2',
            optional_shortcut=optional_shortcut,
            shortcut=self.layers[-2])
        self.layers.append(pool2)
        print("Pooling:", pool2.get_shape())

        # Conv Block 256
        for i in range(num_layers[2]):
            if i < num_layers[2] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(
                inputs=self.layers[-1],
                shortcut=shortcut,
                num_filters=256,
                is_training=self.is_training,
                name=str(i + 1))
            self.layers.append(conv_block)
        pool3 = downsampling(
            self.layers[-1],
            downsampling_type=downsampling_type,
            name='pool3',
            optional_shortcut=optional_shortcut,
            shortcut=self.layers[-2])
        self.layers.append(pool3)
        print("Pooling:", pool3.get_shape())

        # Conv Block 512
        for i in range(num_layers[3]):
            if i < num_layers[3] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(
                inputs=self.layers[-1],
                shortcut=shortcut,
                num_filters=512,
                is_training=self.is_training,
                name=str(i + 1))
            self.layers.append(conv_block)

        # Last pool and then flatten as a vector
        # self.k_pooled = tf.nn.top_k(
        #     tf.transpose(self.layers[-1], [0, 2, 1]),
        #     k=8,
        #     name='k_pool',
        #     sorted=False)[0]
        # print("8-maxpooling:", self.k_pooled.get_shape())
        # self.flatten = tf.reshape(self.k_pooled, (-1, 512 * 8))

        self.Last_pool = tf.nn.max_pool(
            self.layers[-1],
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding="SAME",
            name="Last_pool")
        self.flatten = tf.reshape(self.Last_pool, (batchsize, -1))

        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable(
                'w', [self.flatten.get_shape()[1], 2048],
                initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable(
                'b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.flatten, w) + b
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable(
                'w', [self.fc1.get_shape()[1], 2048],
                initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable(
                'b', [2048], initializer=tf.constant_initializer(1.0))
            out = tf.matmul(self.fc1, w) + b
            self.fc2 = tf.nn.relu(out)

        # TODO: modified the dimensions next lines
        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable(
                'w', [self.fc2.get_shape()[1], input_dim[0] * input_dim[1]],
                initializer=he_normal,
                regularizer=regularizer)
            b = tf.get_variable(
                'b', [input_dim[0] * input_dim[1]],
                initializer=tf.constant_initializer(1.0))
            self.fc3 = tf.matmul(self.fc2, w) + b

        # Calculate Absolute error. I don't think Mean Cross-entropy works well
        with tf.name_scope("loss"):
            self.predictions = tf.reshape(self.fc3,
                                          [-1, input_dim[0], input_dim[1], 1])
            #losses = tf.losses.absolute_difference(self.input_y,
            #                                       self.predictions)
            losses = tf.losses.huber_loss(self.input_y, self.predictions)
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(losses) + sum(regularization_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")