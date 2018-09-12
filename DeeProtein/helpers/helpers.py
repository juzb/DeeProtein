import tensorflow as tf
try:
    import tensorlayer as tl
except ImportError:
    print('Could not import tensorlayer')


def resnet_block(inlayer, channels=[128, 256], pool_dim=2, is_train=True, name='scope', activation=tf.nn.relu):
    """Define a residual block for DeeProtein.

    A residual block consists of two 1d covolutional layers both with a kernel size of 3 and a 1x1 1d convolution.
    Every conv layer is followed by a BatchNorm layer. The input may be pooled (optional).

    Args:
      inlayer: A `tl.layer` object holding the input.
      channels: A `Array` defining the channels.
      pool_dim:  A `int32` defining the pool dims, defaults to 2. May be None (no pooling).
      is_train: A `bool` from which dataset (train/valid) to draw the samples.
      summary_collection: A 'str' object defining the collection to which to attach the summaries of the layers.
        Defaults to `None`.
      name: A 'str' defining the scope to attach to te resBlock. Scopes must be unique in the network.

    Returns:
      A `tl.layer` object holding the Residual Block.
    """
    # calculate the block
    with tf.variable_scope(name) as vs:
        with tf.variable_scope('conv1') as vs:
            conv = tl.layers.Conv1dLayer(inlayer,
                                         act=activation,
                                         shape=[3, channels[0], channels[1]],  # 32 features for each 5x5 patch
                                         stride=1,
                                         padding='SAME',
                                         W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                         W_init_args={},
                                         b_init=tf.constant_initializer(value=0.1),
                                         b_init_args={},
                                         name='cnn_layer')
            _add_var_summary(conv.all_params[-2], 'conv')

            norm = tl.layers.BatchNormLayer(conv, decay=0.9, epsilon=1e-05,
                                            is_train=is_train,
                                            name='batchnorm_layer')
        with tf.variable_scope('conv2') as vs:
            conv = tl.layers.Conv1dLayer(norm,
                                         act=activation,
                                         shape=[3, channels[1], channels[1]*2],  # 32 features for each 5x5 patch
                                         stride=1,
                                         padding='SAME',
                                         W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                         W_init_args={},
                                         b_init=tf.constant_initializer(value=0.1),
                                         b_init_args={},
                                         name='cnn_layer')
            norm = tl.layers.BatchNormLayer(conv, decay=0.9, epsilon=1e-05,
                                            is_train=is_train,
                                            name='batchnorm_layer')
        with tf.variable_scope('1x1') as vs:
            conv = tl.layers.Conv1dLayer(norm,
                                         act=activation,
                                         shape=[1, channels[1]*2, channels[1]],  # 32 features for each 5x5 patch
                                         stride=1,
                                         padding='SAME',
                                         W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                         W_init_args={},
                                         b_init=tf.constant_initializer(value=0.1),
                                         b_init_args={},
                                         name='1x1_layer')
            y = tl.layers.BatchNormLayer(conv, decay=0.9, epsilon=1e-05,
                                         is_train=is_train,
                                         name='batchnorm_layer')
        if pool_dim:
            with tf.variable_scope('pool') as vs:
                in_shape = y.outputs.shape
                y = tl.layers.ReshapeLayer(y, [in_shape[0], in_shape[1],
                                               1, in_shape[2]], name='expand')
                y = tl.layers.PoolLayer(y,
                                        ksize=[1, pool_dim, 1, 1],
                                        strides=[1, pool_dim, 1, 1],
                                        padding='VALID',
                                        name='pool_layer')
                out_shape = [in_shape[0], -1, in_shape[2]]
                y = tl.layers.ReshapeLayer(y, out_shape, name='squeeze')

        with tf.variable_scope('shortcut') as vs:
            # reduce the shortcut
            if pool_dim:
                in_shape = inlayer.outputs.shape
                shortcut = tl.layers.ReshapeLayer(inlayer, [in_shape[0], in_shape[1],
                                               1, in_shape[2]], name='expand')

                shortcut = tl.layers.PoolLayer(shortcut,
                                               ksize=[1, pool_dim, 1, 1],
                                               strides=[1, pool_dim, 1, 1],
                                               padding='VALID', name='pool_layer')
                out_shape = [in_shape[0], -1, in_shape[2]]
                shortcut = tl.layers.ReshapeLayer(shortcut, out_shape, name='squeeze')
            else:
                shortcut = inlayer
            # zero pad the channels
            if channels[0] != channels[1]:
                paddings = [[0,0],
                            [0,0],
                            [0, channels[1]-channels[0]]
                            ]
                shortcut = tl.layers.PadLayer(shortcut, paddings=paddings)

            out = tl.layers.ElementwiseLayer([y, shortcut],
                                             combine_fn=tf.add,
                                             name='merge')
            return out


def _add_var_summary(var, name):
    """Attaches a lot of summaries to a given tensor.

    Args:
      var: A `Tensor`, for which to calculate the summaries.
      name: 'str', the name of the Tensor.
      collection: 'str' the collection to which to add the summary. Defaults to None.
    """
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))


def _variable_on_cpu(name, shape, initializer, trainable):
    """Helper function to get a variable stored on cpu.

    Args:
      name: A 'str' holding the name of the variable.
      shape: An `Array` defining the shape of the Variable. For example: [2,1,3].
      initializer: The `tf.Initializer` to use to initialize the variable.
      trainable: A `bool` stating wheter the variable is trainable or not.

    Returns:
      A `tf.Variable` on CPU.
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


class GO_info():
    """
    Helper class to hold the GO related information.
    """
    def __init__(self, go_file):
        self.go_file = go_file
        self.nclasses = 0
        self.GOs = []
        self.GO_numbers = [] #the integerized GOs
        self.go_counts = []
        self.key2freq = {}
        self.key2id = {}
        self.id2key = {}
        self.get_go_info()

    def _get_GOs(self):
        """Generate the class-dict.
        The class dict stores the link between the index in the one-hot encoding and the class.
        """
        GOs = []
        counts = []
        with open(self.go_file, "r") as go_fobj:
            for line in go_fobj:
                fields = line.strip().split()
                if fields[1].endswith('.csv'):
                    fields[1] = fields[1].rstrip('.csv')
                GOs.append(fields[1])
                counts.append(fields[0])
        return GOs, counts

    def get_go_info(self):
        self.GOs, self.go_counts = self._get_GOs()
        self.nclasses = len(self.GOs)
        self.GO_numbers = list(range(self.nclasses))

        # information for decoding:
        self.key2freq = dict(zip(self.GOs, self.go_counts))
        self.key2id = dict(zip(self.GOs, self.GO_numbers))
        self.id2key = dict(zip(self.GO_numbers, self.GOs))
