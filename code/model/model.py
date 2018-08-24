"""
Model class of dpv2.
This class holds the DeeProtein model and related functions.
This uses a softmax of one negative and one positive neuron per class in the outlayer instead of a sigmoid with just one
"""
import os
import logging
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from helpers import helpers


class Model():
    """
    Model class of dpv2.
    This class holds the DeeProtein model and related functions.
    """
    def __init__(self, FLAGS, go_info, is_train=True):
        self.FLAGS = FLAGS
        self.logger = logging.getLogger('{}.model'.format(self.FLAGS.modelname))
        self.is_train = is_train
        self.go_info = go_info
        self.poolblocks = 0
        self.get_n_poolblocks()
        self.saver = None

    def build_net(self, input):
        """
        Specifies the network architecture.
        :return:
        """

        activation = tf.nn.leaky_relu
        self.logger.info('Using {} activation'.format(activation.__name__))

        n_blocks = self.FLAGS.nblocks # has to be larger than 14, original is 29
        assert n_blocks > 14
        self.logger.info('Number of resnet blocks: {}'.format(n_blocks))

        #get a saver:
        try:
            self.saver = tf.train.Saver()
        except ValueError:
            self.logger.warning('Unable to get train.Saver()')

        # get tensors from input
        seq = input['seq']
        start_pos = input['start_pos']
        length = input['length']
        depth = input['depth']

        # ensure the shapes are correct:
        seq = tf.reshape(seq, [self.FLAGS.batchsize,
                               self.FLAGS.windowlength,
                               20,
                               1], name='initial_reshape')

        seq = tf.transpose(seq, perm=[0, 2, 1, 3])
        with tf.variable_scope('Model') as vs:
            in_layer = tl.layers.InputLayer(seq, name='input_layer')
            with tf.variable_scope('encoder') as vs:
                with tf.variable_scope('embedding') as vs:
                    embedding = tl.layers.Conv2dLayer(in_layer,
                                                      act=activation,
                                                      shape=[20, 1, 1, 64],
                                                      strides=[1, 1, 1, 1],
                                                      padding='VALID',
                                                      W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                      W_init_args={},
                                                      b_init=tf.constant_initializer(value=0.1),
                                                      b_init_args={},
                                                      name='1x1')
                    embedding = tl.layers.BatchNormLayer(embedding, decay=0.9, epsilon=1e-05,
                                                         is_train=self.is_train,
                                                         name='batchnorm_layer')
                    output_shape = embedding.outputs.get_shape().as_list()
                    embedding.outputs = tf.reshape(embedding.outputs,
                                                   shape=[self.FLAGS.batchsize,
                                                          output_shape[2],
                                                          output_shape[3]])
                    helpers._add_var_summary(embedding.outputs, 'conv')

                resnet = helpers.resnet_block(embedding, channels=[64, 128],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res1', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[128, 256],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res2', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[256, 512],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res3', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res4', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res5', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=3, is_train=self.is_train,
                                                   name='res6', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res7', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res8', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res9', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res10', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=None, is_train=self.is_train,
                                                   name='res11', activation=activation)
                resnet = helpers.resnet_block(resnet, channels=[512, 512],
                                                   pool_dim=2, is_train=self.is_train,
                                                   name='res12')
                # here stuff is repetetive:
                for block in range(n_blocks-14): # 12 before, 2 after, current block is always block + 12
                    resnet = helpers.resnet_block(resnet,
                                                  channels=[512, 512],
                                                  pool_dim=None,
                                                  is_train=self.is_train,
                                                  name='res{}'.format((block+13)),
                                                  activation=activation)

                # first nonrepetetive block
                resnet = helpers.resnet_block(resnet,
                                              channels=[512, 1024],
                                              pool_dim=None, is_train=self.is_train,
                                              name='res{}'.format(n_blocks-1),
                                              activation=activation)
                encoder = helpers.resnet_block(resnet,
                                               channels=[1024, 1024],
                                               pool_dim=2, is_train=self.is_train,
                                               name='res{}'.format(n_blocks),
                                               activation=activation)

            with tf.variable_scope('classifier') as vs:
                with tf.variable_scope('out1x1') as vs:
                    classifier = tl.layers.Conv1dLayer(encoder,
                                                        act=tf.identity,
                                                        shape=[1, 1024, 2 * self.go_info.nclasses],
                                                        stride=1,
                                                        padding='SAME',
                                                        W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                                        W_init_args={},
                                                        b_init=tf.constant_initializer(value=0.1),
                                                        b_init_args={},
                                                        name='1x1_layer')
                    # two nodes for each label, one positive, one negative, softmaxed
                    classifier.outputs = tf.reshape(classifier.outputs,
                                     [self.FLAGS.batchsize, self.go_info.nclasses, 2]) # [Batch, Classes, Pos-Neg]
        return classifier

    def get_loss(self, raw_logits, labels, valid_mode=False):
        """Add the loss ops to the current graph.

        Args:
          raw_logits: A `Tensor` holding the activations from the network.
          labels: A `Tensor` holding the one hot encoded ground truth.                  # [Batch, classes]
          valid_mode: A `bool`, define the model in trainings mode or validation mode.

        Returns:
          loss: A `Tensor` object holding the loss as scalar.
          f1_score: A `Tensor` object holding the F1 score.
        """
        name_suffix = '_train'

        if valid_mode:
            name_suffix = '_valid'

        with tf.variable_scope('loss{}'.format(name_suffix)) as vs:

            softmax_logits = tf.nn.softmax(raw_logits, dim=2, name='logits') # [Batch, classes, Pos-Neg]
            softmax_logits = tf.reshape(softmax_logits,
                                             [self.FLAGS.batchsize, self.go_info.nclasses, 2],
                                        name='softmax2predictions')

            softmax_logits = softmax_logits[:, :, 0] # [Batch, classes, Pos]
            # positives
            positive_predictions = tf.cast(tf.greater(softmax_logits, 0.5), dtype=tf.float32)
            true_positive_predictions = tf.multiply(positive_predictions, labels)
            # negatives
            negative_predictions = tf.cast(tf.less(softmax_logits, 0.5), dtype=tf.float32)
            negative_labels = tf.cast(tf.equal(labels, 0), dtype=tf.float32)                # [Batch, classes]
            true_negative_predictions = tf.multiply(negative_predictions, negative_labels)
            false_negative_predictions = tf.multiply(negative_labels, labels)
            false_positive_predictions = tf.multiply(positive_predictions, negative_labels)
            # stats
            nr_pred_positives = tf.reduce_sum(positive_predictions)
            nr_true_positives = tf.reduce_sum(true_positive_predictions)
            nr_true_negatives = tf.reduce_sum(true_negative_predictions)
            nr_false_positives = tf.reduce_sum(false_positive_predictions)
            nr_false_negatives = tf.reduce_sum(false_negative_predictions)
            tpr = tf.divide(nr_true_positives, tf.reduce_sum(labels))
            fdr = tf.divide(nr_false_positives, nr_pred_positives)
            fpr = tf.divide(nr_false_positives, tf.reduce_sum(negative_labels))
            tnr = tf.divide(nr_true_negatives, tf.reduce_sum(negative_labels))

            # accuracy
            f1_score = tf.divide(nr_true_positives*2,
                                 tf.add(tf.add(2*nr_true_positives, nr_false_negatives), nr_false_positives))

            tf.summary.scalar('TPR', tpr)
            tf.summary.scalar('FPR', fpr)
            tf.summary.scalar('FDR', fdr)
            tf.summary.scalar('TNR', tnr)
            tf.summary.scalar('F1', f1_score)
            tf.summary.scalar('avg_pred_positives', tf.divide(nr_pred_positives, self.FLAGS.batchsize))
            tf.summary.scalar('avg_true_positives', tf.divide(nr_true_positives, self.FLAGS.batchsize))

            class_sizes = np.asfarray(list(self.go_info.key2freq.values())) # [classes]
            mean_class_size = np.mean(class_sizes) # [classes]
            pos_weights = mean_class_size / class_sizes # [classes]

            # config.maxClassInbalance prevents too large effective learning rates (i.e. too large gradients)
            assert self.FLAGS.maxclassimbalance >= 1.0

            pos_weights = np.maximum(1.0, np.minimum(self.FLAGS.maxclassimbalance, pos_weights)) # [classes]
            pos_weights = pos_weights.astype(np.float32) # [classes]

            # tile the pos weigths:
            pos_weights = tf.reshape(tf.tile(pos_weights,
                                     multiples=[self.FLAGS.batchsize]),
                                     shape=[self.FLAGS.batchsize, self.go_info.nclasses]) # [batch, classes]
            pos_weights = tf.stack([pos_weights, pos_weights], axis=-1) # [batch, classes, Pos-Neg]

            inverse_labels = tf.cast(tf.equal(labels, 0), dtype=tf.float32) # [batch, classes]

            expanded_labels = tf.stack([labels, inverse_labels], axis=-1) # labels, inverse labels
            expanded_labels = tf.reshape(expanded_labels, shape=[self.FLAGS.batchsize,
                                                                 self.go_info.nclasses, 2]) # [batch, classes, Pos-Neg]

            ce_loss = tf.nn.weighted_cross_entropy_with_logits(logits=raw_logits,
                                                               targets=expanded_labels,
                                                               pos_weight=pos_weights)
            ce_mean = tf.reduce_mean(ce_loss, name='celoss_mean')

            #get the l2 loss on weigths of conv layers and dense layers
            l2_loss = 0
            for w in tl.layers.get_variables_with_name('W_conv1d', train_only=True, printable=False):
                l2_loss += tf.contrib.layers.l2_regularizer(1e-7)(w)
            for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):
                l2_loss += tf.contrib.layers.l2_regularizer(1e-7)(w)
            for w in tl.layers.get_variables_with_name('W', train_only=True, printable=False):
                l2_loss += tf.contrib.layers.l2_regularizer(1e-7)(w)

            loss = ce_mean + l2_loss
            tf.summary.scalar('loss_total', loss)
            tf.summary.scalar('loss_l2', l2_loss)
            tf.summary.scalar('loss_CE', ce_mean)
            self.logger.info("Initialized loss!")
        return loss, f1_score

    def get_opt(self, loss, global_step, vars=[], lr_decay=False, adam=True):
        """Adds an optimizer to the current computational graph.

        Args:
          loss: A `Tensor` 0d, Scalar - The loss to minimize.
          vars: A `list` holding all variables to optimize. If empty all Variables are optmized.
          adam: A `bool` defining whether to use the adam optimizer or not. Defaults to False

        Returns:
          A tf.Optimizer.
        """
        # adaptive learning rate:
        """
        decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
        """
        if lr_decay:
            learning_rate = tf.train.exponential_decay(self.FLAGS.learningrate, global_step,
                                                       decay_steps=100000, decay_rate=0.96, staircase=True)
        else:
            learning_rate = self.FLAGS.learningrate

        if adam:
            if vars:
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                             beta1=0.9, beta2=0.999,
                                             epsilon=self.FLAGS.epsilon,
                                             use_locking=False, name='Adam').minimize(loss, var_list=vars,
                                                                                      global_step=global_step)

            else:
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                             beta1=0.9, beta2=0.999,
                                             epsilon=self.FLAGS.epsilon,
                                             use_locking=False, name='Adam').minimize(loss, global_step=global_step)
            self.logger.info("Initialized ADAM Optimizer!")
        else:
            if vars:
                opt = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                                initial_accumulator_value=0.1,
                                                use_locking=False, name='Adagrad').minimize(loss, var_list=vars,
                                                                                            global_step=global_step)
            else:
                opt = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                                                initial_accumulator_value=0.1,
                                                use_locking=False, name='Adagrad').minimize(loss,
                                                                                            global_step=global_step)
            self.logger.info("Initialized ADAGRAD Optimizer!")

        return opt

    def get_n_poolblocks(self):
        """
        Helper to determine the number of resnet-blocks with pool_dim == 2 and to avoid negative dimesion sizes.
        :return:
        """
        wstart = self.FLAGS.windowlength//(3*2) #as we pool 3 and pool2 before we start defining the actual blocks

        npools = 0
        while wstart > 3:
            wstart = wstart//2
            npools += 1

        self.poolblocks = npools
        self.logger.info('Determined number of pool-blocks to {}'.format(self.poolblocks))

    def save(self, network, session, step):
        """Saves the model into .npz and ckpt files.
        Save the dataset to a file.npz so the model can be reloaded. The model is saved in the
        checkpoints folder under the directory specified in the config dict under the
        summaries_dir key.

        Args:
          network: `tl.layerobj` holding the network.
          session: `tf.session` object from which to save the model.
          step: `int32` the global step of the training process.
        """
        # save model as dict:
        param_save_dir = os.path.join(self.FLAGS.info_path, 'saves')

        # everything but the outlayers
        conv_vars = [var for var in network.all_params
                     if 'dense' and 'outlayer' not in var.name]
        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)
        if conv_vars:
            tl.files.save_npz_dict(conv_vars,
                                   name=os.path.join(param_save_dir,
                                                     'conv_vars.npz'),
                                   sess=session)
        tl.files.save_npz_dict(network.all_params,
                               name=os.path.join(param_save_dir,
                                                 'complete_model.npz'),
                               sess=session)
        self.logger.info('Saved model to .npz !')

        # save also as checkpoint
        if self.saver:
            ckpt_file_path = os.path.join(param_save_dir, 'complete_model.ckpt' )
            self.saver.save(session, ckpt_file_path, global_step=step)
            self.logger.info('Saved model to .ckpt !')
        else:
            self.logger.warning('Could not save model to .cpkt!')

    def restore_weights(self, network, session, nparams=-1):
        """Loads the model up to the last convolutional layer.
        Load the weights for the convolutional layers from a pretrained model.
        Automatically uses the path specified in the config dict under restore_path.

        Args:
          network: `tl.layer` Object holding the network.
          session: `tf.Session` the tensorflow session of which to save the model.
          nparams: 'int' the number of parameters to restore. If -1 all parameters are restored. Defaults to -1
        Returns:
          A tl.Layer object of same size as input, holding the updated weights.
        """
        # check if filepath exists:
        file = os.path.join(self.FLAGS.restorepath, 'conv_vars.npz')
        self.logger.info('Loading {}...'.format(file))
        if not tl.files.file_exists(file):
            self.logger.warning('Loading {} FAILED. File not found.'.format(file))
        # custom load_ckpt op:
        d = np.load(file)

        params = [val[1] for val in sorted(d.items(), key=lambda tup: tup[0])] # changed from int(tup[0])
        params = [p for p in params if not 'outlayer' in p.name] # p is just a numpy array here, doesn't work

        assert nparams >= -1 and nparams != 0, \
            'Please specify number of layers to be restored.' \
            'Must be either a positive int, or -1 (all layers).'

        if nparams > -1:
            params = [p for p in params[:nparams]]
            tl.files.assign_params(session, params, network)
            self.logger.info('Restored weights for the first {} parameters!'.format(nparams))

        elif nparams == -1:
            tl.files.assign_params(session, params, network)
            self.logger.info('Restored weights for the first {} parameters!'.format(len(params)))
        return network

    def load_model_weights(self, network, session):
        """Load the weights for the convolutional layers from a pretrained model.
        If include outlayer is set to True, the outlayers are restored as well,
        otherwise the network is restored without outlayers.

        Args:
          network: `tl.layer` Object holding the network.
          session: `tf.Session` the tensorflow session of whcih to save the model.
          name: 'str', name for the currect network to load. Although optional if multiple
            models are restored, the files are identified by name (optional).
        Returns:
          A tl.Layer object of same size as input, holding the updated weights.
        """
        # check if filepath exists:
        file = os.path.join(self.FLAGS.restorepath, 'complete_model.npz')
        self.logger.info('Loading full model from {}...'.format(file))

        if not tl.files.file_exists(file):
            self.logger.warning('Loading {} FAILED. File not found.'.format(file))

        # standard load cpkt op:
        tl.files.load_and_assign_npz_dict(sess=session, name=file)
        self.logger.info('Restored model weights!')
        return network
