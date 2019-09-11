import tensorflow as tf
from tensorflow import layers


class Model_constructor:
    """Collection of
    fully-connected layers,
    convolution layers,
    batchnorm layers,
    residual blocks,
    stacks of blocks.
    """

    __MODE = None  # 'train' or 'inference'

    __l2loss = None

    _BN = None

    @staticmethod
    def _static_init_(hyps=None):
        """Must be called before graph building."""
        Model_constructor.__l2loss = tf.nn.l2_loss(0.0)
        Model_constructor.__MODE = Model_constructor.cons__MODE_train()
        Model_constructor.__BN = Model_constructor.cons__BN_layers()
        if hyps:
            Model_constructor.__set_hyps(hyps)

    @staticmethod
    def __set_hyps(hyps: dict):
        """Change hyperparameters from default values."""
        for key in hyps.keys():
            if key == 'mode':
                Model_constructor.__set__MODE(hyps[key])
            if key == 'bn':
                Model_constructor.__set__BN(hyps[key])

    @staticmethod
    def conv_kernel_init(shape, name='conv_kernel'):
        initial = tf.truncated_normal(shape, stddev=0.05)
        return tf.Variable(initial, name=name)

    @staticmethod
    def conv_bias_init(shape, name='conv_bias'):
        if len(shape) == 1:
            initial = tf.zeros(shape)
        else:
            initial = tf.zeros([shape[-1]])
        return tf.Variable(initial, name=name)

    @staticmethod
    def batchnorm(conv):
        if Model_constructor.__BN == Model_constructor.cons__BN_layers():
            training = True if Model_constructor.__MODE == Model_constructor.cons__MODE_train() else False
            batch_norm = layers.batch_normalization(conv, training=training)
        elif Model_constructor.__BN == Model_constructor.cons__BN_own0():
            with tf.variable_scope('bn'):
                out_channels = conv.shape.as_list()[-1]

                beta = tf.Variable(tf.zeros([out_channels]), name='beta')
                gamma = tf.Variable(tf.ones([out_channels]), name='gamma')
                pop_mean = tf.Variable(tf.zeros([out_channels]), trainable=False, name='pop_mean')
                pop_var = tf.Variable(tf.ones([out_channels]), trainable=False, name='pop_var')

                mode = Model_constructor.__MODE
                epsilon = 1e-5  #Model_constructor.__BN_EPSILON

                if mode == Model_constructor.cons__MODE_train():
                    batch_mean, batch_var = tf.nn.moments(conv, axes=[0, 1, 2])
                    decay = 0.9 # Model_constructor.__BN_DECAY
                    # train_pop_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                    # train_pop_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                    train_pop_mean = tf.assign(pop_mean, batch_mean)
                    train_pop_var = tf.assign(pop_var, batch_var)
                    with tf.control_dependencies([train_pop_mean, train_pop_var]):
                        batch_norm = tf.nn.batch_normalization(conv, batch_mean, batch_var, beta, gamma, epsilon)
                elif mode == Model_constructor.cons__MODE_inference():
                    batch_norm = tf.nn.batch_normalization(conv, pop_mean, pop_var, beta, gamma, epsilon)
        elif Model_constructor.__BN == Model_constructor.cons__BN_no():
            batch_norm = conv
        return batch_norm

    @staticmethod
    def FC_relu(inpt, output_depth, dropout_p=None, id=None):
        """Fully-connected transition from feature-maps block layer into casual fc layer."""
        with tf.variable_scope('FC_bn_relu_' + str(id[0])):
            inpt_shape = inpt.shape.as_list()
            FC_w = Model_constructor.conv_kernel_init([inpt_shape[1], inpt_shape[2], inpt_shape[3], output_depth])
            Model_constructor.add_l2loss(FC_w)

            FC = tf.nn.conv2d(inpt, FC_w, strides=[1, 1, 1, 1], padding='VALID')
            FC = tf.reshape(FC, [FC.shape[0], FC.shape[3]])
            FC = tf.nn.relu(FC)

            if dropout_p:
                FC = tf.nn.dropout(FC, dropout_p)
        id[0] += 1
        return FC

    @staticmethod
    def FC_bn_relu(inpt, output_depth, dropout_p=None, id=None):
        """Fully-connected transition from feature-maps block layer into casual fc layer."""
        with tf.variable_scope('FC_bn_relu_'+str(id[0])):
            inpt_shape = inpt.shape.as_list()
            FC_w = Model_constructor.conv_kernel_init([inpt_shape[1], inpt_shape[2], inpt_shape[3], output_depth])
            Model_constructor.add_l2loss(FC_w)

            FC = tf.nn.conv2d(inpt, FC_w, strides=[1, 1, 1, 1], padding='VALID')
            FC = tf.reshape(FC, [FC.shape[0], FC.shape[3]])
            FC = Model_constructor.batchnorm(FC)
            FC = tf.nn.relu(FC)

            if dropout_p:
                FC = tf.nn.dropout(FC, dropout_p)
        id[0] += 1
        return FC

    @staticmethod
    def fc_relu(inpt, output_depth, dropout_p=None, id=None):
        """Casual fc layer."""
        with tf.variable_scope('fc_bn_relu_'+str(id[0])):
            fc_w = Model_constructor.conv_kernel_init((inpt.shape.as_list()[-1], output_depth), name='fc_w')
            Model_constructor.add_l2loss(fc_w)

            fc = tf.matmul(inpt, fc_w)
            fc = tf.nn.relu(fc)

            if dropout_p:
                fc = tf.nn.dropout(fc, dropout_p)
        id[0] += 1
        return fc

    @staticmethod
    def fc_bn_relu(inpt, output_depth, dropout_p=None, id=None):
        """Casual fc layer."""
        with tf.variable_scope('fc_bn_relu_'+str(id[0])):
            fc_w = Model_constructor.conv_kernel_init((inpt.shape.as_list()[-1], output_depth), name='fc_w')
            Model_constructor.add_l2loss(fc_w)

            fc = tf.matmul(inpt, fc_w)
            fc = Model_constructor.batchnorm(fc)
            fc = tf.nn.relu(fc)

            if dropout_p:
                fc = tf.nn.dropout(fc, dropout_p)
        id[0] += 1
        return fc

    @staticmethod
    def fc_logits(inpt, output_depth, id):
        with tf.variable_scope('fc_logits_'+str(id[0])):
            fc_w = Model_constructor.conv_kernel_init((inpt.shape.as_list()[-1], output_depth), name='fc_w')
            Model_constructor.add_l2loss(fc_w)
            fc_b = Model_constructor.conv_bias_init([output_depth], name='fc_b')

            fc = tf.matmul(inpt, fc_w) + fc_b
        id[0] += 1
        return fc


    @staticmethod
    def conv_layer(inpt, kernel_shape, stride, id):
        """Conv2d.
        :param inpt: input layer
        :param kernel_shape: shape of convolution filter HWNC
        :param stride: strides of convolution filter HWNC
        :return: output feature map
        """
        with tf.variable_scope('conv_'+str(id[0])):
            kernel = Model_constructor.conv_kernel_init(kernel_shape)
            Model_constructor.add_l2loss(kernel)
            bias = Model_constructor.conv_bias_init(kernel_shape)

            conv = tf.nn.conv2d(inpt, filter=kernel, strides=[1, stride, stride, 1], padding="SAME") + bias
        id[0] += 1
        return conv

    @staticmethod
    def conv_relu_layer(inpt, kernel_shape, stride, id):
        """Conv2d, than relu.
        :param inpt: input layer
        :param kernel_shape: shape of convolution filter HWNC
        :param stride: strides of convolution filter HWNC
        :return: output relu feature map
        """
        with tf.variable_scope('conv_relu_'+str(id[0])):
            kernel = Model_constructor.conv_kernel_init(kernel_shape)
            Model_constructor.add_l2loss(kernel)
            bias = Model_constructor.conv_bias_init(kernel_shape)

            conv = tf.nn.conv2d(inpt, filter=kernel, strides=[1, stride, stride, 1], padding="SAME") + bias
            conv = tf.nn.relu(conv)
        id[0] += 1
        return conv

    @staticmethod
    def conv_bn_layer(inpt, kernel_shape, stride, id):
        """Conv2d, than BN.
        :param inpt: input layer
        :param kernel_shape: shape of convolution filter HWNC
        :param stride: strides of convolution filter HWNC
        :return: output relu feature map
        """
        with tf.variable_scope('conv_bn_'+str(id[0])):
            kernel = Model_constructor.conv_kernel_init(kernel_shape)
            Model_constructor.add_l2loss(kernel)

            conv = tf.nn.conv2d(inpt, filter=kernel, strides=[1, stride, stride, 1], padding="SAME")
            conv = Model_constructor.batchnorm(conv)
        id[0] += 1
        return conv


    @staticmethod
    def conv_bn_relu_layer(inpt, kernel_shape, stride, id):
        """Conv2d, than BN (depending of _MODE), than relu.
        :param inpt: input layer
        :param kernel_shape: shape of convolution filter HWNC
        :param stride: strides of convolution filter HWNC
        :return: output relu feature map
        """
        with tf.variable_scope('conv_bn_relu_'+str(id[0])):
            kernel = Model_constructor.conv_kernel_init(kernel_shape)
            Model_constructor.add_l2loss(kernel)

            conv = tf.nn.conv2d(inpt, filter=kernel, strides=[1, stride, stride, 1], padding="SAME")
            conv = Model_constructor.batchnorm(conv)
            conv = tf.nn.relu(conv)
        id[0] += 1
        return conv

    @staticmethod
    def residual_block_compact(inpt, output_depth, id):
        """Building block for original ResNet 18 and 34 architectures ('compact' means 18 and 34).
        If input_depth != output_depth, this block will realize bottleneck,
        in this case must be: output_depth == input_depth * 2.
        :param inpt: input layer
        :param output_depth: dimension of each conv layer in block (channels, number of feature maps)
        :return: output layer
        """
        with tf.variable_scope('res_block_' + str(id[0])):
            input_depth = inpt.get_shape().as_list()[-1]

            if input_depth == output_depth:
                bottleneck = False
                projection = False
            elif input_depth * 2 == output_depth:
                bottleneck = True
                projection = True
            else:
                raise Exception('Wrong blocks depths for original ResNet18/34. '
                                'Your input_depth = ' + str(input_depth) + ' and output_depth = ' + str(
                    output_depth) + '.')

            # bottleneck by 3x3 conv with stride 2
            block_stride = 2 if bottleneck else 1

            conv_id = [0]
            conv0 = Model_constructor.conv_bn_relu_layer(inpt, [3, 3, input_depth, output_depth], block_stride, conv_id)
            conv1 = Model_constructor.conv_bn_relu_layer(conv0, [3, 3, output_depth, output_depth], 1, conv_id)

            # Option B: identity when dimensions are equal and projection when are not.
            # Dimension are not only equal when it's bottleneck block,
            # so projections with stride 2 in that case
            if projection:
                with tf.variable_scope('projection_shortcat'):
                    inpt = Model_constructor.conv_bn_layer(inpt, [1, 1, input_depth, output_depth], block_stride, [0])

            res = tf.add(conv1, inpt)
        id[0] += 1
        return res

    @staticmethod
    def residual_block_large(inpt, output_depth, id):
        """Building block for original ResNet 50,101,152 architectures.
        Bottleneck in case of: output_depth * 2 == input_depth
        (input_depth is the dimension of the last 1x1conv of previous block (which is an input here)).
        :param inpt: input layer
        :param output_depth: dimension of 1st 1x1conv and 2d 3x3conv in this block; dim 3d 1x1conv is 4*@output_depth
        :return: output layer
        """
        with tf.variable_scope('res_block_' + str(id[0])):
            input_depth = inpt.get_shape().as_list()[-1]

            if input_depth == output_depth:
                # this is only in case of input in 1st residual block (56x56 feature map if 224x224 is image size)
                bottleneck = False
                projection = True
            elif input_depth == output_depth * 4:
                bottleneck = False
                projection = False
            elif input_depth == output_depth * 2:
                assert input_depth % 4 == 0
                assert output_depth * 2 == input_depth
                bottleneck = True
                projection = True
            else:
                raise Exception('Wrong blocks depths for original ResNet50/101/152. '
                                'Your input_depth = ' + str(input_depth) + ' and output_depth = ' + str(
                    output_depth) + '.')

            # bottleneck by 3x3 conv with stride 2
            block_stride = 2 if bottleneck else 1

            conv_id = [0]
            conv0 = Model_constructor.conv_bn_relu_layer(inpt, [1, 1, input_depth, output_depth], block_stride, conv_id)
            conv1 = Model_constructor.conv_bn_relu_layer(conv0, [3, 3, output_depth, output_depth], 1, conv_id)
            conv2 = Model_constructor.conv_bn_relu_layer(conv1, [1, 1, output_depth, output_depth * 4], 1, conv_id)

            # Option B: identity when dimensions are equal and projection when are not.
            if projection:
                with tf.variable_scope('projection_shortcat'):
                    inpt = Model_constructor.conv_bn_layer(inpt, [1, 1, input_depth, output_depth * 4], block_stride)

            res = tf.add(conv2, inpt)
        id[0] += 1
        return res

    @staticmethod
    def residual_block_compact_v2(inpt, output_depth, id):
        with tf.variable_scope('res_block_' + str(id[0])):
            input_depth = inpt.get_shape().as_list()[-1]

            if input_depth == output_depth:
                bottleneck = False
                projection = False
            elif input_depth * 2 == output_depth:
                bottleneck = True
                projection = True
            else:
                raise Exception('Wrong blocks depths for original ResNet18/34. '
                                'Your input_depth = ' + str(input_depth) + ' and output_depth = ' + str(
                    output_depth) + '.')

            # bottleneck by 3x3 conv with stride 2
            block_stride = 2 if bottleneck else 1

            conv_id = [0]
            conv0 = Model_constructor.conv_relu_layer(inpt, [3, 3, input_depth, output_depth], block_stride, conv_id)
            conv1 = Model_constructor.conv_relu_layer(conv0, [3, 3, output_depth, output_depth], 1, conv_id)

            # Option B: identity when dimensions are equal and projection when are not.
            # Dimension are not only equal when it's bottleneck block,
            # so projections with stride 2 in that case
            if projection:
                with tf.variable_scope('projection_shortcat'):
                    inpt = Model_constructor.conv_layer(inpt, [1, 1, input_depth, output_depth], block_stride, [0])

            res = tf.add(conv1, inpt)
        id[0] += 1
        return res

    @staticmethod
    def residual_block_compact_v3(inpt, output_depth, id):
        """Building block for original ResNet 18 and 34 architectures ('compact' means 18 and 34).
        If input_depth != output_depth, this block will realize bottleneck,
        in this case must be: output_depth == input_depth * 2.
        :param inpt: input layer
        :param output_depth: dimension of each conv layer in block (channels, number of feature maps)
        :return: output layer
        """
        with tf.variable_scope('res_block_' + str(id[0])):
            input_depth = inpt.get_shape().as_list()[-1]

            if input_depth == output_depth:
                bottleneck = False
                projection = False
            elif input_depth * 2 == output_depth:
                bottleneck = True
                projection = True
            else:
                raise Exception('Wrong blocks depths for original ResNet18/34. '
                                'Your input_depth = ' + str(input_depth) + ' and output_depth = ' + str(
                    output_depth) + '.')

            # bottleneck by 3x3 conv with stride 2
            block_stride = 1
            if bottleneck:
                inpt = tf.nn.max_pool(inpt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv_id = [0]
            conv0 = Model_constructor.conv_bn_relu_layer(inpt, [3, 3, input_depth, output_depth], block_stride, conv_id)
            conv1 = Model_constructor.conv_bn_relu_layer(conv0, [3, 3, output_depth, output_depth], 1, conv_id)

            # Option B: identity when dimensions are equal and projection when are not.
            # Dimension are not only equal when it's bottleneck block,
            # so projections with stride 2 in that case
            if projection:
                with tf.variable_scope('projection_shortcat'):
                    inpt = Model_constructor.conv_bn_layer(inpt, [1, 1, input_depth, output_depth], block_stride, [0])

            res = tf.add(conv1, inpt)
        id[0] += 1
        return res

    @staticmethod
    def residual_block_compact_v4(inpt, output_depth, id):
        """As keras resnet_v1 application's block:
        projection only conv, last conv_bn without relu, add them than relu
        """
        with tf.variable_scope('res_block_' + str(id[0])):
            input_depth = inpt.get_shape().as_list()[-1]

            if input_depth == output_depth:
                bottleneck = False
                projection = False
            elif input_depth * 2 == output_depth:
                bottleneck = True
                projection = True
            else:
                raise Exception('Wrong blocks depths for original ResNet18/34. '
                                'Your input_depth = ' + str(input_depth) + ' and output_depth = ' + str(
                    output_depth) + '.')

            # bottleneck by 3x3 conv with stride 2
            block_stride = 2 if bottleneck else 1

            conv_id = [0]
            conv0 = Model_constructor.conv_bn_relu_layer(inpt, [3, 3, input_depth, output_depth], block_stride, conv_id)
            conv1 = Model_constructor.conv_bn_layer(conv0, [3, 3, output_depth, output_depth], 1, conv_id)

            # Option B: identity when dimensions are equal and projection when are not.
            # Dimension are not only equal when it's bottleneck block,
            # so projections with stride 2 in that case
            if projection:
                with tf.variable_scope('projection_shortcat'):
                    inpt = Model_constructor.conv_layer(inpt, [1, 1, input_depth, output_depth], block_stride, [0])
            res = tf.add(conv1, inpt)
            res = tf.nn.relu(res)
        id[0] += 1
        return res

    @staticmethod
    def residual_block_compact_v5(inpt, output_depth, id):
        """As keras resnet_v1 application's block:
        projection conv_bn, last conv_bn without relu, add them than relu
        """
        with tf.variable_scope('res_block_' + str(id[0])):
            input_depth = inpt.get_shape().as_list()[-1]

            if input_depth == output_depth:
                bottleneck = False
                projection = False
            elif input_depth * 2 == output_depth:
                bottleneck = True
                projection = True
            else:
                raise Exception('Wrong blocks depths for original ResNet18/34. '
                                'Your input_depth = ' + str(input_depth) + ' and output_depth = ' + str(
                    output_depth) + '.')

            # bottleneck by 3x3 conv with stride 2
            block_stride = 2 if bottleneck else 1

            conv_id = [0]
            conv0 = Model_constructor.conv_bn_relu_layer(inpt, [3, 3, input_depth, output_depth], block_stride, conv_id)
            conv1 = Model_constructor.conv_bn_layer(conv0, [3, 3, output_depth, output_depth], 1, conv_id)

            # Option B: identity when dimensions are equal and projection when are not.
            # Dimension are not only equal when it's bottleneck block,
            # so projections with stride 2 in that case
            if projection:
                with tf.variable_scope('projection_shortcat'):
                    inpt = Model_constructor.conv_bn_layer(inpt, [1, 1, input_depth, output_depth], block_stride, [0])
            res = tf.add(conv1, inpt)
            res = tf.nn.relu(res)
        id[0] += 1
        return res

    @staticmethod
    def stack_of_blocks(inpt, block, output_depth, i, id):
        """Stack of blocks with the same feature-map size.
        Bottleneck between stacks in case of different stacks dimensions (depth of stack).
        :param inpt: input layer
        :param block: Callable, (residual) block of layers
        :param output_depth: dimension of each block (channels, number of feature maps)
        :param i: number of @block in stack
        :return: output layer
        """
        with tf.variable_scope('stack_of_blocks_'+str(id[0])):
            block_id = [0]
            for k in range(i):
                inpt = block(inpt, output_depth, block_id)
        id[0] += 1
        return inpt

    @staticmethod
    def get__l2loss():
        return Model_constructor.__l2loss

    @staticmethod
    def get__MODE():
        return Model_constructor.__MODE

    @staticmethod
    def __set__MODE(mode):
        assert mode == Model_constructor.cons__MODE_train() or mode == Model_constructor.cons__MODE_inference()
        Model_constructor.__MODE = mode

    @staticmethod
    def cons__MODE_train():
        return 'train'

    @staticmethod
    def cons__MODE_inference():
        return 'inference'

    @staticmethod
    def __set__BN(bn):
        assert bn == Model_constructor.cons__BN_layers() or bn == Model_constructor.cons__BN_own0() \
               or bn == Model_constructor.cons__BN_no()
        Model_constructor.__BN = bn

    @staticmethod
    def cons__BN_layers():
        return 'layers'

    @staticmethod
    def cons__BN_own0():
        return 'own0'

    @staticmethod
    def cons__BN_no():
        return 'no'

    @staticmethod
    def add_l2loss(w):
        if Model_constructor.__MODE == Model_constructor.cons__MODE_train():
            Model_constructor.__l2loss += tf.nn.l2_loss(w)
