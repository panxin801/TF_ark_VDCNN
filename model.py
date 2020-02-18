import tensorflow as tf
import numpy as np
import math

# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)


def Deconvolutional_Block_Last(inputs, num_filters, output_shape, name,
                               is_training):
    print("-" * 20)
    print("Deconvolutional Block", str(num_filters), name)
    print("-" * 20)
    current_name = int(name) - 1
    with tf.variable_scope("deconv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            filter_shape = [
                1,
                inputs.get_shape()[2], num_filters,
                inputs.get_shape()[3]
            ]
            if current_name == 0 and i == 0:
                filter_shape = [1, 20, num_filters, inputs.get_shape()[3]]
                stride = [1, 1, 1, 1]
                padding = "VALID"
            else:
                stride = [1, 1, 1, 1]
                padding = "SAME"
            with tf.variable_scope("deconv1d_%s" % str(i)):
                W = tf.get_variable(
                    name='W',
                    shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv2d_transpose(
                    inputs, W, output_shape, strides=stride, padding=padding)
                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    momentum=0.997,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Deconv1D:", inputs.get_shape())
    print("-" * 20)
    return inputs


def Deconvolutional_Block_256(inputs, num_filters, output_shape, name,
                              is_training, block_num):
    print("-" * 20)
    print("Deconvolutional Block", str(num_filters), name)
    print("-" * 20)
    current_name = int(name) - 1
    with tf.variable_scope("deconv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            filter_shape = [
                1,
                inputs.get_shape()[2], num_filters,
                inputs.get_shape()[3]
            ]
            # if current_name == block_num - 1 and i == 2 - 1:
            if current_name == 0 and i == 0:
                filter_shape = [1, 6, num_filters, inputs.get_shape()[3]]
                stride = [1, 1, 1, 1]
                padding = "VALID"
            else:
                stride = [1, 1, 1, 1]
                padding = "SAME"
            with tf.variable_scope("deconv1d_%s" % str(i)):
                W = tf.get_variable(
                    name='W',
                    shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv2d_transpose(
                    inputs, W, output_shape, strides=stride, padding=padding)
                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    momentum=0.997,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Deconv1D:", inputs.get_shape())
    print("-" * 20)
    return inputs


def Deconvolutional_Block_128(inputs, num_filters, output_shape, name,
                              is_training, block_num):
    print("-" * 20)
    print("Deconvolutional Block", str(num_filters), name)
    print("-" * 20)
    current_name = int(name) - 1
    with tf.variable_scope("deconv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            filter_shape = [
                1,
                inputs.get_shape()[2], num_filters,
                inputs.get_shape()[3]
            ]
            if current_name == 0 and i == 0:
                filter_shape = [1, 11, num_filters, inputs.get_shape()[3]]
                stride = [1, 1, 1, 1]
                padding = "VALID"
            else:
                stride = [1, 1, 1, 1]
                padding = "SAME"
            with tf.variable_scope("deconv1d_%s" % str(i)):
                W = tf.get_variable(
                    name='W',
                    shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv2d_transpose(
                    inputs, W, output_shape, strides=stride, padding=padding)
                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    momentum=0.997,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Deconv1D:", inputs.get_shape())
    print("-" * 20)
    return inputs


def Deconvolutional_Block(inputs, num_filters, output_shape, name, is_training,
                          block_num):
    print("-" * 20)
    print("Deconvolutional Block", str(num_filters), name)
    print("-" * 20)
    current_name = int(name) - 1
    with tf.variable_scope("deconv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            filter_shape = [
                1,
                inputs.get_shape()[2], num_filters,
                inputs.get_shape()[3]
            ]
            if current_name == 0 and i == 0:
                filter_shape = [1, 5, num_filters, inputs.get_shape()[3]]
                stride = [1, 1, 1, 1]
                padding = "VALID"
            else:
                stride = [1, 1, 1, 1]
                padding = "SAME"
            with tf.variable_scope("deconv1d_%s" % str(i)):
                W = tf.get_variable(
                    name='W',
                    shape=filter_shape,
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv2d_transpose(
                    inputs, W, output_shape, strides=stride, padding=padding)
                inputs = tf.layers.batch_normalization(
                    inputs=inputs,
                    momentum=0.997,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Deconv1D:", inputs.get_shape())
    print("-" * 20)
    return inputs


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
            ksize=[1, 1, 2, 1],  # [1,1,3,1]->[1,1,2,1]
            strides=[1, 1, 2, 1],
            padding='SAME',
            name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(
            inputs=shortcut,
            filters=shortcut.get_shape()[2],
            kernel_size=1,
            strides=2,
            padding='sAME',
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
        padding='SAME',  # "VALID"->"SAME"
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
        self.deconvOutShape = []

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
            # inputs = tf.nn.relu(inputs)
        print("First Conv", inputs.get_shape())
        self.layers.append(inputs)
        self.deconvOutShape.append(inputs.get_shape())

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
        self.deconvOutShape.append(conv_block.get_shape())
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
        self.deconvOutShape.append(conv_block.get_shape())
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
        self.deconvOutShape.append(conv_block.get_shape())
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
        self.deconvOutShape.append(conv_block.get_shape())

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
        self.layers.append(self.Last_pool)

        # The following parts are deconv block
        # Deconv Block 512
        for i in range(num_layers[3]):
            deconv_block = Deconvolutional_Block(
                inputs=self.layers[-1],
                num_filters=512,
                output_shape=self.deconvOutShape[4],
                is_training=self.is_training,
                name=str(i + 1),
                block_num=num_layers[3])
            self.layers.append(deconv_block)

        # Deconv Block 256
        for i in range(num_layers[2]):
            deconv_block = Deconvolutional_Block_256(
                inputs=self.layers[-1],
                num_filters=256,
                output_shape=self.deconvOutShape[3],
                is_training=self.is_training,
                name=str(i + 1),
                block_num=num_layers[2])
            self.layers.append(deconv_block)
        # pool3 = downsampling(
        #     self.layers[-1],
        #     downsampling_type=downsampling_type,
        #     name='pool3',
        #     optional_shortcut=optional_shortcut,
        #     shortcut=self.layers[-2])
        # self.layers.append(pool3)
        # print("Pooling:", pool3.get_shape())

        # Deconv Block 128
        for i in range(num_layers[1]):
            deconv_block = Deconvolutional_Block_128(
                inputs=self.layers[-1],
                num_filters=128,
                output_shape=self.deconvOutShape[2],
                is_training=self.is_training,
                name=str(i + 1),
                block_num=num_layers[1])
            self.layers.append(deconv_block)

        # Deconv Block 64
        for i in range(num_layers[0]):
            deconv_block = Deconvolutional_Block_Last(
                inputs=self.layers[-1],
                num_filters=64,
                output_shape=self.deconvOutShape[1],
                is_training=self.is_training,
                name=str(i + 1))
            self.layers.append(deconv_block)

        # Last Deconv Layer
        with tf.variable_scope("Last_Deconv"):
            filter_shape = [1, 3, 1, 64]
            W = tf.get_variable(
                name='W_1',
                shape=filter_shape,
                initializer=he_normal,
                regularizer=regularizer)
            output = tf.nn.conv2d_transpose(
                self.layers[-1],
                W, [batchsize, input_dim[0], input_dim[1], 1],
                strides=[1, 1, 1, 1],
                padding="SAME")
            # inputs = tf.nn.relu(inputs)
        print("Last Conv", output.get_shape())
        self.layers.append(output)
        # End of deconv block

        # fc1
        # with tf.variable_scope('fc1'):
        #     w = tf.get_variable(
        #         'w', [self.flatten.get_shape()[1], 2048],
        #         initializer=he_normal,
        #         regularizer=regularizer)
        #     b = tf.get_variable(
        #         'b', [2048], initializer=tf.constant_initializer(1.0))
        #     out = tf.matmul(self.flatten, w) + b
        #     self.fc1 = tf.nn.relu(out)

        with tf.name_scope("loss"):
            # self.predictions = tf.reshape(self.layers[-1],
            #                               [-1, input_dim[0], input_dim[1], 1])
            self.predictions = self.layers[-1]
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


# Define downconv
def downconv(x,
             output_dim,
             kwidth=5,
             pool=2,
             init=None,
             uniform=False,
             bias_init=None,
             name='downconv'):
    """ Downsampled convolution 1d """
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable(
            'W',
            [kwidth, kwidth, x.get_shape()[-1], output_dim],
            initializer=w_init)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, pool, 1], padding='SAME')
        if bias_init is not None:
            b = tf.get_variable('b', [output_dim], initializer=bias_init)
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        else:
            conv = tf.reshape(conv, conv.get_shape())
        # reshape back to 1d
        # conv = tf.reshape(
        #     conv,
        #     conv.get_shape().as_list()[:2] + [conv.get_shape().as_list()[-1]])
        return conv


def prelu(x, name='prelu', ref=False):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # make one alpha per feature
        alpha = tf.get_variable(
            'alpha',
            in_shape[-1],
            initializer=tf.constant_initializer(0.),
            dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
        if ref:
            # return ref to alpha vector
            return pos + neg, alpha
        else:
            return pos + neg


def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)


class ANFCN():
    def __init__(self, input_dim, batchsize, is_ref=True, do_prelu=False):
        kwidth = 5  # 31->5
        skips = []
        # g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        g_enc_depths = [16, 32, 32, 64, 64, 128]
        self.input_x = tf.placeholder(
            tf.float32, [batchsize, input_dim[0], input_dim[1], 1],
            name="noise_feat")
        self.input_y = tf.placeholder(
            tf.float32, [batchsize, input_dim[0], input_dim[1], 1],
            name="clean_feat")
        self.is_training = tf.placeholder(tf.bool)

        h_i = self.input_x
        with tf.variable_scope('Encode'):
            # Encode-Decode struct to be built is shaped:
            # enc ~ [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
            # dec ~ [2048, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 1]
            # FIRST ENCODER
            alphas = []
            for layer_idx, layer_depth in enumerate(g_enc_depths):
                bias_init = None
                h_i_dwn = downconv(
                    h_i,
                    layer_depth,
                    kwidth=kwidth,
                    init=tf.truncated_normal_initializer(stddev=0.02),
                    bias_init=bias_init,
                    name='enc_{}'.format(layer_idx))
                if is_ref:
                    print('Downconv {} -> {}'.format(h_i.get_shape(),
                                                     h_i_dwn.get_shape()))
                h_i = h_i_dwn
                if layer_idx < len(g_enc_depths) - 1:
                    if is_ref:
                        print('Adding skip connection downconv '
                              '{}'.format(layer_idx))
                    # store skip connection last one is not stored cause it's the code
                    skips.append(h_i)
                if do_prelu:
                    if is_ref:
                        print('-- Enc: prelu activation --')
                    h_i = prelu(
                        h_i, ref=is_ref, name='enc_prelu_{}'.format(layer_idx))
                    if is_ref:
                        # split h_i into its components
                        alpha_i = h_i[1]
                        h_i = h_i[0]
                        alphas.append(alpha_i)
                else:
                    if is_ref:
                        print('-- Enc: leakyrelu activation --')
                    h_i = leakyrelu(h_i)

            # if z_on:
            #     # random code is fused with intermediate representation
            #     z = make_z([
            #         segan.batch_size,
            #         h_i.get_shape().as_list()[1], segan.g_enc_depths[-1]
            #     ])
            #     h_i = tf.concat([z, h_i], 2)

            # SECOND DECODER (reverse order)
            g_dec_depths = g_enc_depths[:-1][::-1] + [1]
            if is_ref:
                print('g_dec_depths: ', g_dec_depths)
            for layer_idx, layer_depth in enumerate(g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                # out_shape = [
                #     h_i_dim[0], h_i_dim[1], h_i_dim[2] * 2, layer_depth
                # ]
                if layer_idx < len(g_dec_depths) - 1:
                    ssha = skips[-(layer_idx + 1)].shape.as_list()
                else:
                    ssha = self.input_x.shape.as_list()
                out_shape = [h_i_dim[0], h_i_dim[1], ssha[2], layer_depth]
                bias_init = None
                # deconv
                if is_ref:
                    print('-- Transposed deconvolution type --')
                # if layer_idx in [1]:
                #     h_i_dcv = deconv(
                #         h_i,
                #         out_shape,
                #         padding="VALID",
                #         kwidth=kwidth,
                #         dilation=3,
                #         init=tf.truncated_normal_initializer(stddev=0.02),
                #         bias_init=bias_init,
                #         name='dec_{}'.format(layer_idx))
                # else:
                h_i_dcv = deconv(
                    h_i,
                    out_shape,
                    kwidth=kwidth,
                    dilation=2,
                    init=tf.truncated_normal_initializer(stddev=0.02),
                    bias_init=bias_init,
                    name='dec_{}'.format(layer_idx))
                if is_ref:
                    print('Deconv {} -> {}'.format(h_i.get_shape(),
                                                   h_i_dcv.get_shape()))
                h_i = h_i_dcv
                if layer_idx < len(g_dec_depths) - 1:
                    if do_prelu:
                        if is_ref:
                            print('-- Dec: prelu activation --')
                        h_i = prelu(
                            h_i,
                            ref=is_ref,
                            name='dec_prelu_{}'.format(layer_idx))
                        if is_ref:
                            # split h_i into its components
                            alpha_i = h_i[1]
                            h_i = h_i[0]
                            alphas.append(alpha_i)
                    else:
                        if is_ref:
                            print('-- Dec: leakyrelu activation --')
                        h_i = leakyrelu(h_i)
                    # fuse skip connection
                    skip_ = skips[-(layer_idx + 1)]
                    if is_ref:
                        print('Fusing skip connection of '
                              'shape {}'.format(skip_.get_shape()))
                    h_i = tf.concat([h_i, skip_], 3)
                else:
                    if is_ref:
                        print('-- Dec: tanh activation --')
                    h_i = tf.tanh(h_i)

            wave = h_i
            if is_ref and do_prelu:
                print('Amount of alpha vectors: ', len(alphas))
            gen_wave_summ = tf.summary.histogram('gen_wave', wave)
            if is_ref:
                print('Amount of skip connections: ', len(skips))
                print('Last wave shape: ', wave.get_shape())
                print('*************************')
            generator_built = True
            # ret feats contains the features refs to be returned
            ret_feats = [wave]
            # if z_on:
            #     ret_feats.append(z)
            # if is_ref and do_prelu:
            #     ret_feats += alphas

        with tf.name_scope("loss"):
            # self.predictions = tf.reshape(self.layers[-1],
            #                               [-1, input_dim[0], input_dim[1], 1])
            self.predictions = wave
            # losses = tf.losses.absolute_difference(self.input_y,
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


def deconv(x,
           output_shape,
           padding="SAME",
           kwidth=5,
           dilation=2,
           init=None,
           uniform=False,
           bias_init=None,
           name='deconv1d'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    out_channels = output_shape[-1]
    assert len(input_shape) >= 4
    # reshape the tensor to use 2d operators
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        # filter shape: [kwidth, output_channels, in_channels]
        W = tf.get_variable(
            'W', [kwidth, kwidth, out_channels, in_channels],
            initializer=w_init)
        deconv = tf.nn.conv2d_transpose(
            x,
            W,
            output_shape=output_shape,
            strides=[1, 1, dilation, 1],
            padding=padding)
        if bias_init is not None:
            b = tf.get_variable(
                'b', [out_channels], initializer=tf.constant_initializer(0.))
            deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        else:
            deconv = tf.reshape(deconv, deconv.get_shape())
        return deconv

