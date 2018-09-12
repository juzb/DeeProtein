import tensorflow as tf
import logging


class Dataset_Gen:
    """Reads  a csv containing name, sequence and GO-terms for one protein per line (from datatpreprocessing)
    gives batches to model efficiently"""

    def __init__(self, FLAGS, go_info, train=True, data='seq', mask_width=1, seq=None):
        """Sets attributes of the Dataset Generator"""
        self.logger = logging.getLogger('{}.dataset_gen'.format(FLAGS.modelname))

        if train:
            self.data_path = FLAGS.traindata
        else:
            self.data_path = FLAGS.validdata

        self.nepochs = FLAGS.nepochs
        self.batchsize = FLAGS.batchsize
        self.windowlength = FLAGS.windowlength
        self.mask_width = mask_width # just needed for masked datasets

        self.go_info = go_info
        self.wt_seq_str = None

        # hashtable implementation for aa-lookup
        self.keys = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        self.depth = len(self.keys)
        self.values = list(range(len(self.keys)))

        self.table_aa2id = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(self.keys, self.values), -1)

        # hashtable implementation for go-lookup
        self.table_go2id = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(self.go_info.GOs, self.go_info.GO_numbers), -1)

        # dataset definition
        if data == 'seq':
            self.dataset = (tf.data.TextLineDataset(self.data_path)  # Read text file
                               .map(self.decode_csv_seq))  # Transform each elem by applying decode_csv_seq fn
            self.dataset = self.dataset.repeat(1)  # Number of epochs per dataset.

        elif data == 'mut':
            # Save the wt sequence to be able to generate the full sequences from the mutations
            with open(self.data_path, 'r') as ifile:
                _ = ifile.readline()
                self.wt_seq_str = [ifile.readline().strip()]
            # self.logger.debug('WT-SEQ: \n{}\n\n'.format(self.wt_seq_str[0]))
            self.dataset = (tf.data.TextLineDataset(self.data_path)  # Read text file
                               .skip(2) # skip header line and sequence line
                               .map(self.decode_csv_mut))  # Transform each elem by applying decode_csv_mut fn
            self.dataset = self.dataset.repeat(1)  # Number of epochs per dataset.

        elif data == 'mask':
            # get variables ready
            self.pos = tf.cast(tf.Variable(0), dtype=tf.int32, name='position') # currently masked position

            if seq:
                self.wt_seq_str = seq.split(';')[0]
                label_str = seq.split(';')[1]
            else:
                with open(self.data_path, 'r') as ifile:
                    file = ifile.readline().strip()
                    label_str = file.split(';')[1]

                self.wt_seq_str = file.split(';')[2]
            self.wt_seq_ten = tf.constant([self.wt_seq_str])

            self.label = []
            for x in label_str.split(','):
                self.label.append(self.labels2tensor(tf.constant(x))[1])

            seq_str = tf.sparse_tensor_to_dense(tf.string_split(self.wt_seq_ten, ''), '', name='seq_as_str')

            self.one_hot_seq, self.start_pos, self.seq_length = self.seq2tensor(seq_str)

            pos_tensor = tf.range(start=-1,limit= len(self.wt_seq_str))

            self.dataset = tf.data.Dataset.from_tensor_slices(pos_tensor).map(self.mask)

        elif data == 'none':
            self.logger.debug('Dataset generator just used for its functions')

        else:
            self.logger.debug('unknown type of data: {}'.format(data))

        if data != 'none':
            self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batchsize))

        self.type = 'Train'
        if not train:
            self.type = 'Valid'

    def uniprot_csv_parser(self, line):
        """(adapted from datapreprocessing.py) Parser for one line of the uniprot.csv.

        The uniprot csv file is in the following syntax:
        name;seq;GO1,GO2,...

        Args:
          line: 'str' the input line

        returns:
          name: 'str' the name from uniprot
          seq: 'str' the sequence from uniprot
          GOs: list of 'str', all relevant GOs

        """
        fields = tf.sparse_tensor_to_dense(tf.string_split(tf.expand_dims(line, 0), ';'), '', name='fields')

        name = fields[0, 0]
        seq = fields[0, 1]
        labels = fields[0, 2]
        GOs = tf.reshape(tf.sparse_tensor_to_dense(
                tf.string_split(
                    tf.expand_dims(labels, 0),
                    ','),
                '', name='labels'),
            shape=[-1], name='GOs')
        return name, seq, GOs

    def seq2tensor(self, seq):
        """Takes a sequence and encodes it onehot.
        Args:
          seq: `str Tensor`, The sequence to encode

        Returns:
          padded_seq_matrix: A `Tensor` holding the one-hot encoded sequence.
          start_pos: A `Tensor` holding the start pos.
          length: A `Tensor` holding the length.
        """

        # make the string tensor accessible:
        windowlength = tf.constant([self.windowlength], name='windowlength')
        start_pos = tf.constant(0, dtype=tf.int32, shape=[1], name='start_pos')  # sequence at the beginning of the box

        # translate the string seq to int and give it a shape
        seq_in_int = self.table_aa2id.lookup(seq[0, :], name='seq_in_int')
        seq_length = tf.size(seq_in_int, name='seq_length')
        seq_in_int = tf.reshape(seq_in_int, [seq_length], name='seq_in_int_shaped')

        def pad():  # in case the sequence is shorter than the windowlength
            to_pad = tf.subtract(
                tf.zeros(shape=windowlength, dtype=tf.int32),
                tf.ones(shape=windowlength, dtype=tf.int32))
            # for dynamic padding with -1 (-1 is translated in a vector of zeros by tf.one_hot
            sliced_zeros = tf.slice(to_pad,
                                    tf.constant([0]),
                                    tf.reshape(tf.subtract(windowlength, seq_length), [1], name='-1padding'))
            return tf.concat([seq_in_int, sliced_zeros], axis=0, name='padded_seq')

        def slice():  # in case the sequence is longer than the windowlength
            return tf.reshape(tf.slice(seq_in_int, [0], windowlength), windowlength, name='sliced_seq')

        # decide wether to pad or to slice
        seq_correct_len = tf.cond(pred=tf.less_equal(tf.reshape(seq_length, []), tf.reshape(windowlength, [])),
                                  true_fn=pad,
                                  false_fn=slice,
                                  name='seq_corr_length')
        # make the final one-hot encoded sequence with a fixed length and shape
        one_hot_seq = tf.expand_dims(tf.one_hot(seq_correct_len, self.depth, name='one_hot_seq'), -1)

        return one_hot_seq, start_pos, seq_length  # true length 1 based

    def labels2tensor(self, labels):
        """
        Convert the labels to tensors.
        """
        labels_in_int = self.table_go2id.lookup(labels, name='labels_in_int')
        labels_one_hot = tf.one_hot(labels_in_int, self.go_info.nclasses, name='labels_one_hot')
        labels_vector = tf.reduce_sum(labels_one_hot, axis=0, name='labels_vector')
        return labels_vector, labels_in_int

    def decode_csv_seq(self, line):
        """Applied to all lines in the csv-file from which the dataset is built. Used for Sequence datasets:
        <Name>;<Sequence>;<GO1>,<GO2>,..."""
        # get relevant fields from the csv
        _, seq, labels = self.uniprot_csv_parser(line)

        seq = tf.expand_dims(seq, 0)
        # encode the sequence and its other features
        seq = tf.sparse_tensor_to_dense(tf.string_split(seq, ''), '', name='seq_as_str')

        padded_oh_seq, start_pos, length = self.seq2tensor(seq)
        features = {'seq':          padded_oh_seq,
                    'start_pos':    start_pos,
                    'length':       length,
                    'depth':        self.depth}

        # encode the labels in a vector ('multi-hot')
        encoded_labels, _ = self.labels2tensor(labels)

        return [features, encoded_labels]

    def mask(self, pos):
        """
        Mask the given sequence position.
        """
        def f():
            """
            Helper for tf.cond
            """
            masked = tf.zeros(shape=[self.mask_width, self.depth, 1])
            before = self.one_hot_seq[:pos, :]
            after = self.one_hot_seq[(pos + self.mask_width):, :]
            full = tf.concat([before, masked, after], axis=0)
            return full

        def t():
            """
            Helper for tf.cond
            """
            return self.one_hot_seq

        seq = tf.cond(pred=tf.equal(pos, -1),
                      true_fn=t,
                      false_fn=f,
                      name='check_first_seq')

        features = {'seq': seq,
                    'depth': self.depth,
                    'start_pos': 0,
                    'length': 1000,
                    'mask_width': self.mask_width}

        return [features, pos]
