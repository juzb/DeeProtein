"""
The Class controlling all the stuff in the model.
"""
import os
import time
import logging
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from goatools.obo_parser import GODag
from helpers.dataset_gen import Dataset_Gen
from helpers.helpers import *
from helpers.prettyplotter import PrettyPlotter
from datapreprocessing.datapreprocessing import DataPreprocessor
import importlib.util
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd


class Controller:
    """
    Manages the model and all code to be executed around it
    """
    def __init__(self, FLAGS):
        # the controller will hold the flags, the data, the session, the plotter, the model and logger
        self.FLAGS = FLAGS
        self.logger = logging.getLogger('{}.controller'.format(self.FLAGS.modelname))
        self.logger.info('\n============================================================\n\n\n\nNew call:\n')

        # import the model specified in the FLAGS
        spec = importlib.util.spec_from_file_location("model.model", self.FLAGS.model_to_use)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        global Model
        Model = mod.Model

        # define attributes to be filled later:
        self.model = None
        self.train_dsgen = None
        self.valid_dsgen = None
        self.iterator = None
        self.session = None
        self.graph = None
        self.pp = None
        self.GODag = None

        self.go_info = GO_info(self.FLAGS.gofile)

        # log success.
        self.logger.info('Controlla Object initialized with FLAGS:\n{}'.format(
            str(FLAGS).replace('Namespace(', '').replace(')', '').replace(', ', ',\n')
        ))

    def initialize(self):
        """
        Prepare everything for Model to be run. E.g. call a tf.Session, initialize the dataset_gens
        """

        if self.FLAGS.allowsoftplacement == 'True':
            config = tf.ConfigProto(allow_soft_placement=True)
            self.logger.debug('Set configProto to allow_soft_placement.')
        else:
            config = tf.ConfigProto(allow_soft_placement=False)

        # allow growth to surveil the consumed GPU memory
        config.gpu_options.allow_growth = True
        # get a graph:
        self.graph = tf.Graph()

        # open a session:
        self.session = tf.Session(config=config, graph=self.graph)

        self.logger.debug('Invoked Session. Got empty Graph.')
        self.logger.info('Further initialized Controller for running the model.')

    def train(self):
        """Start the training process for the model.

        Runs training and validation ops with preferences specified in the config.JSON.
        """
        self.initialize()
        with self.graph.as_default():
            # Invoke the dataset generators and the preprocessing pipeline:
            self.train_dsgen = Dataset_Gen(FLAGS=self.FLAGS, go_info=self.go_info, train=True, data='seq')

            self.valid_dsgen = Dataset_Gen(FLAGS=self.FLAGS, go_info=self.go_info, train=False, data='seq')

            handle = tf.placeholder(tf.string, shape=[])  # this is feeded whether to use valid or train dataset
            self.iterator = tf.data.Iterator.from_string_handle(string_handle=handle,
                                                                output_types=self.train_dsgen.dataset.output_types,
                                                                output_shapes=self.train_dsgen.dataset.output_shapes,
                                                                )
            # helper iterators that feed self.iterator, these are reinitializable
            train_iterator = tf.data.Iterator.from_structure(self.train_dsgen.dataset.output_types,
                                                             self.train_dsgen.dataset.output_shapes)
            valid_iterator = tf.data.Iterator.from_structure(self.train_dsgen.dataset.output_types,
                                                             self.train_dsgen.dataset.output_shapes)
            # define the init_ops:
            train_init_op = train_iterator.make_initializer(self.train_dsgen.dataset)
            train_init_op_once = tf.group(self.train_dsgen.table_aa2id.init,
                                          self.train_dsgen.table_go2id.init)

            valid_init_op = valid_iterator.make_initializer(self.valid_dsgen.dataset)
            valid_init_op_once = tf.group(self.valid_dsgen.table_aa2id.init,
                                          self.valid_dsgen.table_go2id.init)

            train_handle = self.session.run(train_iterator.string_handle())
            valid_handle = self.session.run(valid_iterator.string_handle())

            self.logger.info('Invoked Dataset_Generators')

            self.model = Model(FLAGS=self.FLAGS, go_info=self.go_info, is_train=True)
            self.logger.info('Invoked Model.')

            # Now put everything together
            with self.graph.device('/device:GPU:0'):
                # ^this assumes CUDA_VISIBLE_DEVICES is used so that just one gpu is visible
                # get a global step to control the learning rate decay:
                global_step = tf.Variable(0, trainable=False)

                # define the model
                batch_samples, batch_labels = self.iterator.get_next()
                batch_prediction = self.model.build_net(batch_samples)

                # Invoke the Event writers:
                self.train_writer = tf.summary.FileWriter(self.FLAGS.info_path + '/tensorboard/train',
                                                          self.session.graph)
                self.eval_writer = tf.summary.FileWriter(self.FLAGS.info_path + '/tensorboard/valid')

                # feed dicts:
                train_fd = {handle: train_handle}
                valid_fd = {handle: valid_handle}

                # add the loss and optimizer
                loss, f1_internal = self.model.get_loss(batch_prediction.outputs, batch_labels, valid_mode=False)
                train_params = batch_prediction.all_params
                optimizer = self.model.get_opt(loss, global_step=global_step, lr_decay=False, adam=True,
                                               vars=train_params)

                # merge the summaries:
                summaries = tf.summary.merge_all()

                # do the inits
                self.session.run(train_init_op_once, feed_dict=train_fd)
                self.session.run(valid_init_op_once, feed_dict=valid_fd)
                self.session.run(tf.global_variables_initializer())
                self.logger.debug('Initialized Variables.')

                # restore model if wanted
                if self.FLAGS.restore == 'True':
                    batch_prediction = self.model.restore_weights(batch_prediction, self.session, nparams=-1)  # 136)

                if self.FLAGS.reload_checkpoint == 'True':
                    batch_prediction = self.model.load_model_weights(batch_prediction, self.session)

                self.logger.debug('Total number of trainable Variables: {}'
                                  .format(len(batch_prediction.all_params)))

                if self.FLAGS.print_num_params == 'True':
                    self.logger.debug('Total number of trainable parameters: {:,}'.
                                      format(sum([tf.size(x).eval(session=self.session)
                                                  for x in batch_prediction.all_params]
                                                 )))
                self.logger.info('Starting training. Epochs: {}'.format(self.FLAGS.nepochs))
                starting_time = time.time()

                for epoch in range(self.FLAGS.nepochs):
                    # train
                    self.session.run(train_init_op)

                    self.logger.debug('Ran train_init_op, initialized variables')
                    step = 0
                    while True:
                        try:
                            step, summary, _, _ = self.session.run([global_step,
                                                                    summaries,
                                                                    optimizer,
                                                                    batch_prediction.outputs],
                                                                   feed_dict=train_fd)

                            self.train_writer.add_summary(summary, step)

                            if step % 100 == 0:
                                self.logger.debug('T - Epoch {}, Batch {} done.'.format(epoch, step))

                            if step % self.FLAGS.valid_after == 0:  # Validation during epoch: Shorter
                                self.logger.debug('Valid during one epoch')
                                f1_collection = []
                                validation_predictions = []
                                validation_labels = []
                                true_positives = []
                                true_negatives = []
                                self.session.run(valid_init_op)
                                self.logger.debug('Ran valid_init_op')
                                valid_step = step  # to make it appear on the same height in the tensorboard
                                max_valid_batches = self.FLAGS.early_valid_size
                                while True:
                                    try:
                                        if valid_step % 100 == 0:
                                            self.logger.debug('V* - Epoch {}, Batch {} done.'.format(epoch, valid_step))
                                        valid_step += 1
                                        max_valid_batches -= 1
                                        summary, f1, predicted_labels, labels = self.session.run([summaries,
                                                                                                  f1_internal,
                                                                                                  batch_prediction.outputs,
                                                                                                  batch_labels],
                                                                                                 feed_dict=valid_fd)
                                        # log the stuff for tensorboard
                                        self.eval_writer.add_summary(summary, valid_step)
                                        validation_predictions.append(predicted_labels)
                                        predicted_labels = np.greater(predicted_labels, 0.5)
                                        validation_labels.append(labels)
                                        true_positives.append(np.asarray(validation_labels)[0, :, :]
                                                              * np.asarray(predicted_labels)[:, :, 0])
                                        true_negatives.append((1 - np.asarray(validation_labels)[0, :, :])
                                                              * (1 - np.asarray(predicted_labels)[:, :, 0]))

                                        f1_collection.append(f1)
                                        if max_valid_batches == 0:
                                            raise tf.errors.OutOfRangeError(node_def=None, op=None,
                                                                            message='exceeded maximum valid batches')

                                    except tf.errors.OutOfRangeError:
                                        accuracy = float((np.sum(true_positives) +
                                                          np.sum(true_negatives)) / np.size(true_positives))
                                        self.logger.info('Epoch {}/{}: Finished validation during epoch with avg.'
                                                         'F1-Score: {} on {} batches'.format(epoch + 1,
                                                                                             self.FLAGS.nepochs,
                                                                                             float(np.mean(np.asarray(
                                                                                                 f1_collection),
                                                                                                 keepdims=False)),
                                                                                             len(f1_collection)))
                                        self.logger.info('Epoch {} finished with accuracy: {}'.format(epoch, accuracy))
                                        break

                                validation_predictions = np.concatenate(validation_predictions, axis=0)
                                validation_labels = np.concatenate(validation_labels, axis=0)
                                self.write_predictions(validation_predictions, validation_labels)

                                self.get_metrics()
                                self.plot(step=step, early=True)

                                # save model
                                self.model.save(network=batch_prediction, session=self.session, step=global_step)

                            if self.FLAGS.valid_after == 0:
                                self.logger.info('Performed the validation, closing now.')
                                return
                        except tf.errors.OutOfRangeError:
                            self.logger.info('Epoch {}/{}: Finished training.'.format(epoch + 1, self.FLAGS.nepochs))
                            break  # we are done with the epoch

                    # validation
                    self.logger.debug('Valid after epoch')
                    f1_collection = []
                    validation_predictions = []
                    validation_labels = []
                    true_positives = []
                    true_negatives = []

                    self.session.run(valid_init_op, feed_dict=valid_fd)
                    self.logger.debug('Ran valid_init_op')
                    valid_step = step  # to make it appear on the same height in the tensorboard
                    while True:
                        try:
                            if valid_step % 100 == 0:
                                self.logger.debug('V - Epoch {}, Batch {} done.'.format(epoch, valid_step))
                            valid_step += 1
                            _, summary, f1, predicted_labels, labels = self.session.run([global_step,
                                                                                         summaries,
                                                                                         f1_internal,
                                                                                         batch_prediction.outputs,
                                                                                         batch_labels],
                                                                                        feed_dict=valid_fd)

                            # log the stuff for tensorboard
                            self.eval_writer.add_summary(summary, valid_step)
                            validation_predictions.append(predicted_labels)
                            predicted_labels = np.greater(predicted_labels, 0.5)
                            validation_labels.append(labels)
                            true_positives.append(np.asarray(validation_labels)[0, :, :]
                                                  * np.asarray(predicted_labels)[:, :, 0])
                            true_negatives.append((1 - np.asarray(validation_labels)[0, :, :])
                                                  * (1 - np.asarray(predicted_labels)[:, :, 0]))

                            f1_collection.append(f1)

                        except tf.errors.OutOfRangeError:
                            self.logger.debug('Out of range: Valid done.')
                            accuracy = float((np.sum(true_positives)
                                              + np.sum(true_negatives)) / np.size(true_positives))
                            self.logger.info('Epoch {}/{}: Finished validation during epoch with avg.'
                                             'F1-Score: {} on {} batches'.format(epoch + 1,
                                                                                 self.FLAGS.nepochs,
                                                                                 float(
                                                                                     np.mean(np.asarray(f1_collection),
                                                                                             keepdims=False)),
                                                                                 len(f1_collection)))

                            self.logger.info('Epoch {} finished with accuracy: {}'.format(epoch, accuracy))
                            break

                    validation_predictions = np.concatenate(validation_predictions, axis=0)
                    self.logger.debug('Concatenated predictions')
                    validation_labels = np.concatenate(validation_labels, axis=0)
                    self.logger.debug('Concatenated labels')
                    self.write_predictions(validation_predictions, validation_labels)
                    self.logger.debug('Wrote predicitions')

                    self.get_metrics()
                    self.logger.debug('Got metrics')
                    self.plot(step=valid_step, early=False)
                    self.logger.debug('Plotted.')

                    # save model
                    self.model.save(network=batch_prediction, session=self.session, step=global_step)
                    self.logger.debug('Saved model.')

                self.logger.info('Finished training. {} epochs in {}.'.format(self.FLAGS.nepochs,
                                                                              str(time.time() - starting_time)))

    def plot(self, step, early):
        """
        Initialize a PrettyPlotter instance and do all plots
        :param step: I
        :return:
        """

        if not self.pp:
            self.pp = PrettyPlotter(self.FLAGS, self.go_info)
            os.makedirs(os.path.join(self.FLAGS.info_path, 'plots'), exist_ok=True)

        self.pp.plot_all(step, early)
        self.logger.info('Plotted metrics plots.')

    def write_predictions(self, predictions, labels):
        """
        Write the predictions and the labels to a file.
        :param predictions: nd-Array, the predictions
        :param labels: nd-Array, the ground truth
        """
        os.makedirs(os.path.join(self.FLAGS.info_path, 'metrics'), exist_ok=True)

        prediction_dump = os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'), 'raw_predictions.npy')
        labels_dump = os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'), 'raw_labels.npy')
        # dump as np arrays:
        np.save(prediction_dump, predictions)
        np.save(labels_dump, labels)
        self.logger.debug('Dumped predictions and labels.')

    def get_metrics(self):
        """
        Reads predictions back from disk and calculates the most important metrics.
        Writes them to metrics.npz
        :return:
        """

        softmax_outlayer = True  # set to false when using sigmoid outlayer
        detailed_plots = (self.FLAGS.detailed_plots.lower() == 'true')

        self.logger.info('Calculating Metrics.')
        prediction_dump = os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'), 'raw_predictions.npy')
        labels_dump = os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'), 'raw_labels.npy')

        # first load the metrics from the npy dumps in the metrics directory
        predictions = np.load(prediction_dump)
        labels = np.load(labels_dump)
        self.logger.debug('Loaded metrics dumps.')

        # Try to get a GODag:
        if not self.GODag:
            try:
                self.GODag = GODag(self.FLAGS.godagfile, optional_attrs=['relationship'])
                self.logger.debug('Loaded GO-DAG.')
            except OSError:
                self.logger.warning('No GO-Dag file found. Information may be incomplete when calculating metrics.')
                self.GODag = None

        # calculate class-wise metrics. By the idx in the array.
        # get a dir where to save all this:
        os.makedirs(os.path.join(self.FLAGS.info_path, 'metrics'), exist_ok=True)
        per_class_dir = os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'), 'per_class_dumps')
        os.makedirs(per_class_dir, exist_ok=True)

        no_f1 = []

        for i in range(self.go_info.nclasses):
            class_labels = labels[:, i]
            if softmax_outlayer:
                class_predictions = predictions[:, i, 0]
            else:
                class_predictions = predictions[:, i]
            # now get the ROC:
            class_fpr_arr, class_tpr_arr, _ = roc_curve(
                y_true=np.reshape(class_labels, (-1)),
                y_score=np.reshape(class_predictions, (-1)))
            class_roc_auc = float(auc(class_fpr_arr, class_tpr_arr))

            # precision/recall curve:
            class_precision_arr, class_recall_arr, _ = precision_recall_curve(
                y_true=np.reshape(class_labels, (-1)), probas_pred=np.reshape(class_predictions, (-1)))
            class_precision_recall_auc = float(auc(class_recall_arr, class_precision_arr))

            # Term wise Fmax score
            class_fmax = (2 * class_precision_arr * class_recall_arr / (class_precision_arr + class_recall_arr)).max()

            # F1-Score
            class_positive_predictions = np.greater(class_predictions, 0.5)
            class_positive_predictions = class_positive_predictions.astype(float)

            # F1-Score w/o any avg
            class_tp_predictions = np.sum(np.multiply(class_positive_predictions, class_labels))
            class_precision = np.sum(class_tp_predictions) / np.sum(class_positive_predictions)
            class_recall = np.sum(class_tp_predictions) / np.sum(class_labels)
            class_false_negatives = np.sum(np.multiply(class_labels, (1 - class_positive_predictions)))
            fnr = class_false_negatives / np.sum(class_labels)

            try:
                assert np.sum(class_positive_predictions) > 0
                class_f1_hand = float(2 * class_precision * class_recall / (class_precision + class_recall))

            except AssertionError:
                no_f1.append(self.go_info.id2key[i])
                class_f1_hand = np.nan

            # store them in the class_metrics_dict
            per_class_metric = {}
            per_class_metric['n_samples'] = float(np.sum(class_labels))  # these are the valid samples!
            per_class_metric['tp_predictions'] = class_tp_predictions if not np.isnan(class_tp_predictions) else 0.0
            per_class_metric['ROC_auc'] = class_roc_auc if not np.isnan(class_roc_auc) else 0.0
            per_class_metric['precision'] = class_precision if not np.isnan(class_precision) else 0.0
            per_class_metric['recall'] = class_recall if not np.isnan(class_recall) else 0.0
            per_class_metric['fnr'] = fnr if not np.isnan(fnr) else 1.0
            per_class_metric['f1'] = class_f1_hand if not np.isnan(class_f1_hand) else 0.0
            per_class_metric['fmax'] = class_fmax if not np.isnan(class_fmax) else 0.0
            per_class_metric['id'] = i
            per_class_metric['GO'] = self.go_info.id2key[i]
            per_class_metric['precision_recall_auc'] = float(class_precision_recall_auc) \
                if not np.isnan(class_precision_recall_auc) else 0.0
            if detailed_plots:
                per_class_metric['fpr_arr'] = class_fpr_arr
                per_class_metric['tpr_arr'] = class_tpr_arr
                per_class_metric['precision_arr'] = class_precision_arr
                per_class_metric['recall_arr'] = class_recall_arr

            # add the actual name of the GO-Term:
            try:
                per_class_metric['name'] = self.GODag[self.go_info.id2key[i]].name
                per_class_metric['level'] = self.GODag[self.go_info.id2key[i]].level

            except TypeError:
                per_class_metric['name'] = None
                per_class_metric['level'] = None

            # save this in the per_class_folder as npz:
            per_class_metrics_dump = os.path.join(per_class_dir, '{}_metrics.npy'.format(per_class_metric['GO']))
            np.savez(per_class_metrics_dump, **per_class_metric)

        if no_f1:
            self.logger.warning(
                "No f1-scores because of missing positive predictions for:\n{}".format(", ".join(no_f1))
            )
        self.logger.info('Done with per_class metrics!')

        # now get over-all metrics:

        if softmax_outlayer:
            predictions = predictions[:, :, 0]

        fpr, tpr, thresholds = roc_curve(y_true=np.ravel(labels),
                                         y_score=np.ravel(predictions))
        roc_auc = auc(fpr, tpr)

        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true=np.ravel(labels),
                                                                       probas_pred=np.ravel(predictions))
        precision_recall_auc = auc(recall_arr, precision_arr)

        fmax = (2 * precision_arr * recall_arr / (precision_arr + recall_arr)).max()

        # dump this as .npy files:
        overall_metrics_dump = os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'), 'metrics.npy')
        save_var_dict = {'fpr': fpr,
                         'tpr': tpr,
                         'roc_auc': roc_auc,
                         'precision_arr': precision_arr,
                         'recall_arr': recall_arr,
                         'precision_recall_auc': precision_recall_auc,
                         'fmax': fmax}
        np.savez(overall_metrics_dump, **save_var_dict)
        self.logger.info('Done with over-all metrics. Dump saved in npz_dict {}'.format(overall_metrics_dump))

    def evaluate_masked_set(self, filepath=None):
        """
        Runs the model for occlusion based sensitivity analysis on sequences with masking
        Needs the sequence in a .txt specified in the valid path -v.
        """
        assert self.FLAGS.batchsize == 1  # Otherwise the last positions won't be evaluated
        os.makedirs(os.path.join(self.FLAGS.info_path, 'metrics/all_diffs'), exist_ok=True)
        os.makedirs(os.path.join(self.FLAGS.info_path, 'aa_resolution'), exist_ok=True)
        os.makedirs(os.path.join(self.FLAGS.info_path, 'metrics/seqs/'), exist_ok=True)

        self.logger.info('Writing the results of the sensitivity analysis '
                         'to {}'.format(os.path.join(self.FLAGS.info_path, 'aa_resolution')))

        self.initialize()
        self.pp = PrettyPlotter(self.FLAGS, self.go_info)
        if filepath:
            p = filepath
        else:
            p = self.FLAGS.validdata
        # Invoke the dataset generators and the preprocessing pipeline:
        seqs = []
        ids = []
        gos = []
        dis = []
        secs = []
        chains = []
        with open(p, 'r') as ifile:
            for line in ifile:
                cont = line.strip().split(';')
                if len(cont) == 6:
                    ids.append(cont[0])
                    chains.append(cont[1])
                    gos.append(cont[2].split(','))
                    seqs.append(cont[3] + ';' + cont[2])
                    secs.append(cont[4])
                    dis.append(cont[5])
                else:
                    self.logger.debug('Problem with line: [{}]'.format(line.strip()))

        num_datapoint = self.FLAGS.num_datapoint_offset

        self.dsgens = {}

        max_number_of_datapoints = self.FLAGS.nepochs
        with self.graph.as_default():
            # dummy dsgen, will be overwritten but has correct types and shapes
            self.dsgen = Dataset_Gen(FLAGS=self.FLAGS,
                                     go_info=self.go_info,
                                     train=False,
                                     data='mask',
                                     mask_width=1,
                                     seq=seqs[0])

            self.logger.info('Starting to score {} datasets.'.format(max_number_of_datapoints))
            with self.graph.device('/device:GPU:0'):
                # ^this assumes CUDA_VISIBLE_DEVICES is used so that just one gpu is visible

                handle = tf.placeholder(tf.string, shape=[])  # this is feeded whether to use the different datasets
                self.iterator = tf.data.Iterator.from_string_handle(string_handle=handle,
                                                                    output_types=self.dsgen.dataset.output_types,
                                                                    output_shapes=self.dsgen.dataset.output_shapes,
                                                                    )
                self.logger.info('Invoked Dataset_Generators')

                self.model = Model(FLAGS=self.FLAGS, go_info=self.go_info, is_train=False)
                self.logger.info('Invoked Model.')

                # define the model
                batch_samples, batch_labels = self.iterator.get_next()
                batch_prediction = self.model.build_net(batch_samples)

                self.session.run(tf.global_variables_initializer())

                # load the model
                batch_prediction = self.model.load_model_weights(batch_prediction, self.session)
                batch_prediction = batch_prediction.outputs

                for n_seq in range(len(seqs)):

                    all_predictions = {}
                    rel_predictions = {}
                    rel_diffs = {}
                    all_positions = {}
                    mean = {}
                    stdev = {}
                    all_diffs = {}
                    class_ints = None

                    mw = self.FLAGS.mask_width

                    out_path = os.path.join(self.FLAGS.info_path,
                                            'aa_resolution/masked_{}_{}_{}.txt'.format(ids[n_seq],
                                                                                       chains[n_seq],
                                                                                       mw))
                    if os.path.exists(out_path):
                        self.logger.info(
                            'Entry with ID {} {} was already evaluated for mw {} '
                            'in this information directory.'.format(
                                ids[n_seq], chains[n_seq], mw))
                        continue
                    self.logger.debug('Evaluation of {} {} with mask width {} now '
                                      '(Entry {} of {}, {:.1f} %).'.format(
                        ids[n_seq],
                        chains[n_seq],
                        mw,
                        n_seq,
                        len(range(int(self.FLAGS.mask_width / 2))) * len(ids),
                        (100 * n_seq / len(ids))
                    ))

                    # New Dataset generator with everything:
                    self.dsgen = Dataset_Gen(FLAGS=self.FLAGS,
                                             go_info=self.go_info,
                                             train=False,
                                             data='mask',
                                             mask_width=mw,
                                             seq=seqs[n_seq])

                    iterator = tf.data.Iterator.from_structure(
                        self.dsgen.dataset.output_types,
                        self.dsgen.dataset.output_shapes)

                    str_handle = self.session.run(iterator.string_handle())
                    fd = {handle: str_handle}

                    # define the init_ops:
                    init_op = iterator.make_initializer(self.dsgen.dataset)
                    init_op_once = tf.group(self.dsgen.table_aa2id.init,
                                            self.dsgen.table_go2id.init)
                    self.session.run(init_op_once)
                    self.session.run(init_op)
                    seq_str = self.dsgen.wt_seq_str

                    if not class_ints:
                        class_ints = list(set(self.dsgen.label.eval(session=self.session)))
                        if -1 in class_ints:
                            class_ints.remove(-1)
                        gos = [self.dsgen.dict_go2id[ci] for ci in class_ints]

                        assert (len(gos) == len(class_ints))

                    if len(class_ints) == 0:
                        self.logger.warning('No GO terms found that can be evaluated for '
                                            '{} {}.'.format(ids[n_seq], chains[n_seq]))
                        continue

                    with open(os.path.join(os.path.join(self.FLAGS.info_path, 'metrics'),
                                           'seqs/wt_seq_{}_{}.txt'.format(ids[n_seq], chains[n_seq])),
                              'w') as ofile:
                        ofile.write(seq_str)

                    all_predictions[mw] = []
                    all_positions[mw] = []
                    count = 0
                    while True:
                        try:
                            # predictions, labels = self.session.run([batch_prediction.outputs,
                            predictions, labels = self.session.run([batch_prediction,
                                                                    batch_labels],
                                                                   feed_dict=fd)

                            all_predictions[mw].append(predictions)
                            all_positions[mw].append(labels)
                            count += 1

                        except tf.errors.OutOfRangeError:
                            break
                    self.logger.debug('Evaluated {} different input sequences.'.format(count))

                    # evaluate:
                    all_positions[mw] = np.asarray(all_positions[mw])
                    all_positions[mw] = np.concatenate(all_positions[mw], axis=0)

                    all_predictions[mw] = np.asarray(all_predictions[mw])
                    all_predictions[mw] = np.concatenate(all_predictions[mw], axis=0)

                    rel_predictions[mw] = []
                    rel_diffs[mw] = []
                    mean[mw] = []
                    stdev[mw] = []
                    all_diffs[mw] = []
                    print(all_predictions[mw].shape)

                    with open(out_path, 'w') as ofile:
                        gos = [go + n_p for go in gos for n_p in ['_+', '_-']]

                        ofile.write('Pos\tAA\tsec\tdis\t{}\n'.format('\t'.join(gos)))

                        seq_chars = [c for c in seq_str]
                        sec_chars = [c for c in secs[n_seq]]
                        dis_chars = [c for c in dis[n_seq]]

                        wt_preds = []
                        for c in class_ints:
                            for n_p in range(2):
                                wt_preds.append(str(all_predictions[mw][0, c, n_p]))

                        ofile.write('-1\twt\twt\twt\t{}\n'.format('\t'.join(wt_preds)))
                        for i in range(len(seq_str)):
                            content = [str(all_positions[mw][i + 1])]
                            content.append(seq_chars[i])
                            try:
                                content.append(sec_chars[i])
                            except:
                                content.append(' ')
                            try:
                                content.append(dis_chars[i])
                            except:
                                content.append(' ')

                            # starting gos here:
                            for c in class_ints:
                                for n_p in range(2):
                                    content.append(str(all_predictions[mw][i + 1, c, n_p]))

                            ofile.write('{}\n'.format('\t'.join(content)))
                    num_datapoint += 1

        self.logger.info('Done.\n\n\n')

    def init_for_infer(self):
        """
        Initializes the state of the controller for inference
        """
        assert self.FLAGS.batchsize == 1
        self.initialize()
        with self.graph.as_default():
            # Invoke the dataset generators and the preprocessing pipeline:
            self.model = Model(FLAGS=self.FLAGS, go_info=self.go_info, is_train=False)
            self.logger.info('Invoked Model.')

            # Now put everything together
            with self.graph.device('/device:GPU:0'):
                # ^this assumes CUDA_VISIBLE_DEVICES is used so that just one gpu is visible
                self.dsgen_for_translation = Dataset_Gen(FLAGS=self.FLAGS, go_info=self.go_info)
                init_op = tf.group(self.dsgen_for_translation.table_aa2id.init,
                                   self.dsgen_for_translation.table_go2id.init)
                self.inf_seq_str = tf.placeholder(tf.string, [])
                seq_str = tf.expand_dims(self.inf_seq_str, 0)
                seq_ten = tf.sparse_tensor_to_dense(tf.string_split(seq_str, ''), '', name='seq_as_str')
                _padded_oh_seq, _start_pos, _length = self.dsgen_for_translation.seq2tensor(seq_ten)

                inf_batch_samples = {'seq': _padded_oh_seq,
                                     'start_pos': _start_pos,
                                     'length': _length,
                                     'depth': self.FLAGS.depth}
                self.inf_batch_prediction = self.model.build_net(inf_batch_samples)

                # do the inits
                self.session.run(tf.global_variables_initializer())

                # load the model
                raw_logits = self.model.load_model_weights(self.inf_batch_prediction, self.session)

                self.inf_batch_prediction = tf.nn.softmax(raw_logits.outputs, dim=2, name='logits')

        self.session.run(init_op)
        if not self.GODag:
            try:
                self.GODag = GODag(self.FLAGS.godagfile, optional_attrs=['relationship'])
                self.logger.debug('Loaded GO-DAG.')
            except OSError:
                self.logger.warning('No GO-Dag file found. Information may be incomplete when calculating metrics.')
                self.GODag = None

    def infer(self, seq):
        """
        Run inference on a single sequence
        :param seq: Sequence as a string in one-letter code
        :return: Formatted output
        """
        # feed dicts:
        fd = {self.inf_seq_str: seq}
        predictions = self.session.run([self.inf_batch_prediction],
                                       feed_dict=fd)

        predictions = np.asarray(predictions)
        if self.FLAGS.print_all == 'False':
            t_predictions = np.greater(predictions, 0.5)[0, 0, :, 0]
            prediction_ints = np.where(t_predictions)[0]
        else:
            t_predictions = predictions[0, 0, :, 0]
            order = t_predictions.argsort()

            prediction_ints = np.flip(order, axis=0)

        labels = []
        names = []
        first = True
        output = '\nPredicted labels:\n-----------------\n\nGO-Term\t\tScore\tExplanation\n'

        for pred_int in prediction_ints:
            labels.append(self.go_info.id2key[pred_int])
            try:
                names.append(self.GODag[labels[-1]].name)
            except AttributeError:
                names.append('')
            if first and predictions[0, 0, pred_int, 0] < 0.5:
                output += 50 * '-' + '\n'
                first = False
            if self.FLAGS.hide_zeros == 'True':
                if '{:.3f}'.format(predictions[0, 0, pred_int, 0]) == '0.000':
                    break
            output += '{}\t{:.3f}\t{}\n'.format(labels[-1], predictions[0, 0, pred_int, 0], names[-1])

        return output

    def interactive_inference(self):
        """
        Runs interactive inference mode on CLI
        :return:
        """
        print('\n' * 50)
        print('Close Inference mode with CTRL+C.\n\nScores below 0.5 are interpreted as negative predictions.\n\n')
        self.protocol = open(os.path.join(self.FLAGS.info_path, 'inference_protocol.txt'), 'a')
        self.protocol.write('\n\n\nNew Call at {}:\n'.format(datetime.datetime.now()))
        dpp = DataPreprocessor('', func_only=True)
        while True:
            sequence = input('Enter Sequence for Classification:\n\n')
            self.protocol.write('\nInput:\n\n{}\n\n'.format(sequence))
            if dpp._valid_seq(sequence):
                labels = self.infer(sequence)
                self.prlog(labels)
            else:
                self.prlog('\nThe given sequence was not valid. '
                           'It either contained forbidden characters or was shorter than {} characters.\n'
                           .format(dpp.minlength))
            self.protocol.flush()

    def prlog(self, content):
        """
        Writes content from the interactive inference mode to the protocol file (SIDE EFFECT) and to STDOUT
        :param content:
        :return:
        """
        self.protocol.write('{}\n'.format(content))
        print(content)

    def eval_test(self):
        """
        Runs all the tf model components necessary for model evaluation on a test set
        :return:
        """
        self.initialize()
        os.makedirs(os.path.join(self.FLAGS.info_path, 'metrics'), exist_ok=True)

        with self.graph.as_default():
            # Invoke the dataset generators and the preprocessing pipeline:
            self.dsgen = Dataset_Gen(FLAGS=self.FLAGS, go_info=self.go_info, train=False, data='seq')

            handle_ph = tf.placeholder(tf.string, shape=[])  # this is feeded whether to use valid or train dataset
            self.iterator = tf.data.Iterator.from_string_handle(string_handle=handle_ph,
                                                                output_types=self.dsgen.dataset.output_types,
                                                                output_shapes=self.dsgen.dataset.output_shapes,
                                                                )
            # helper iterators that feed self.iterator, these are reinitializable
            iterator = tf.data.Iterator.from_structure(self.dsgen.dataset.output_types,
                                                       self.dsgen.dataset.output_shapes)

            # define the init_ops:
            init_op = iterator.make_initializer(self.dsgen.dataset)
            init_op_once = tf.group(self.dsgen.table_aa2id.init,
                                    self.dsgen.table_go2id.init)

            handle = self.session.run(iterator.string_handle())

            self.logger.info('Invoked Dataset_Generators')

            self.model = Model(FLAGS=self.FLAGS, go_info=self.go_info, is_train=True)
            self.logger.info('Invoked Model.')

            # Now put everything together
            with self.graph.device('/device:GPU:0'):
                # ^this assumes CUDA_VISIBLE_DEVICES is used so that just one gpu is visible

                # define the model
                batch_samples, batch_labels = self.iterator.get_next()
                batch_ids = batch_samples['id']
                batch_prediction = self.model.build_net(batch_samples)

                fd = {handle_ph: handle}

                # do the inits
                self.session.run(init_op_once, feed_dict=fd)
                self.session.run(tf.global_variables_initializer())
                self.logger.debug('Initialized Variables.')
                # restore model if wanted

                model = self.model.load_model_weights(batch_prediction, self.session)

                softmax_logits = tf.nn.softmax(model.outputs, dim=2, name='logits')  # [Batch, classes, Pos-Neg]
                softmax_logits = tf.reshape(softmax_logits,
                                            [self.FLAGS.batchsize, self.go_info.nclasses, 2],
                                            name='softmax2predictions')

                self.logger.debug('Total number of trainable Variables: {}'
                                  .format(len(model.all_params)))

                if self.FLAGS.print_num_params == 'True':
                    self.logger.debug('Total number of trainable parameters: {:,}'.
                                      format(sum([tf.size(x).eval(session=self.session)
                                                  for x in model.all_params]
                                                 )))
                # start evaluation
                self.logger.info('Starting evaluation')
                starting_time = time.time()

                self.session.run(init_op)

                self.logger.debug('Ran init_op, initialized variables')

                # validation
                self.logger.debug('Valid after epoch')
                validation_predictions = []
                validation_labels = []
                true_positives = []
                true_negatives = []
                validation_ids = []

                step = 0
                while True:
                    try:
                        if step % 100 == 0:
                            self.logger.debug('E - Batch {} done.'.format(step))
                        step += 1
                        predicted_labels, labels, ids = self.session.run([softmax_logits,  # batch_prediction.outputs,
                                                                          batch_labels,
                                                                          batch_ids],
                                                                         feed_dict=fd)

                        validation_predictions.append(predicted_labels)

                        predicted_labels = np.greater(predicted_labels, 0.5)
                        validation_labels.append(labels)
                        true_positives.append(np.asarray(validation_labels)[0, :, :]
                                              * np.asarray(predicted_labels)[:, :, 0])
                        true_negatives.append((1 - np.asarray(validation_labels)[0, :, :])
                                              * (1 - np.asarray(predicted_labels)[:, :, 0]))
                        validation_ids.append(ids)

                    except tf.errors.OutOfRangeError:
                        break

                accuracy = float((np.sum(true_positives) + np.sum(true_negatives)) / np.size(true_positives))

                self.logger.info('Test evaluation of {} batches finished with accuracy: {}'.format(step, accuracy))

                # write the validation metrics to a metrics.csv clean the collectors
                validation_predictions = np.concatenate(validation_predictions, axis=0)
                validation_labels = np.concatenate(validation_labels, axis=0)
                validation_ids = np.concatenate(validation_ids, axis=0)
                self.write_predictions(validation_predictions, validation_labels, validation_ids)

            self.logger.info('Finished evaluation in {}.'.format(str(time.time() - starting_time)))

    def test(self):
        """
        Evaluates the model on a test dataset specified by the -validdata flag
        :return:
        """
        self.logger.info('Using {} for testing.'.format(self.FLAGS.validdata))
        os.makedirs(os.path.join(self.FLAGS.info_path, 'metrics'), exist_ok=True)

        try:
            ds_stat_df = pd.read_csv(self.FLAGS.dataset_statistics, index_col=0)
            ds_stat_df.sort_index()
        except FileNotFoundError:
            self.logger.exception("-ds flag not set to an appropriate training dataset statics file.")
            raise
        # Try to get a GODag:
        if not self.GODag:
            try:
                self.GODag = GODag(self.FLAGS.godagfile, optional_attrs=['relationship'])
                self.logger.debug('Loaded GO-DAG.')
            except OSError:
                self.logger.warning('No GO-Dag file found. Information may be incomplete when calculating metrics.')
                self.GODag = None

        self.eval_test()
        self.logger.debug('Evaluation complete.')
        self.get_metrics()
        self.logger.debug('Got metrics.')

        if not self.pp:
            self.pp = PrettyPlotter(self.FLAGS, self.go_info)
            self.pp.read_per_class_metrics()
            self.logger.debug('Got per class metrics.')
        else:
            self.logger.debug('Found existing pretty plotter!!!')

        head = list(self.pp.per_class_metrics_list[0][0])

        head = head + ['nGO_children']
        head = head + ['avg_n_gos', 'avg_len', 'n_orgs']
        # if go_wise:
        #     head = head + list(list(go_wise.values())[0].keys())

        head = sorted(head, key=lambda x: x.lower())
        head.remove('GO')
        head = ['GO'] + head
        if 'fpr_arr' in head:
            head.remove('fpr_arr')
        if 'precision_arr' in head:
            head.remove('precision_arr')
        if 'recall_arr' in head:
            head.remove('recall_arr')
        if 'tpr_arr' in head:
            head.remove('tpr_arr')

        self.logger.debug('Head: {}'.format(', '.join(head)))

        # write initial table.csv:
        self.tablepath = os.path.join(self.FLAGS.info_path, 'full_table.csv')
        with open(self.tablepath, 'w') as ofile:
            ofile.write('{}\n'.format('; '.join(head)))

            for comp_entry in self.pp.per_class_metrics_list:
                for go_entry in comp_entry:
                    go = str(go_entry['GO'])
                    content = []
                    for category in head:
                        if category == 'nGO_children':
                            content.append(len(self.GODag.query_term(go).get_all_children()))
                        elif category in ['n_samples', 'avg_n_gos', 'avg_len', 'n_orgs']:
                            content.append(ds_stat_df[category].get(go, default="NaN"))
                        else:
                            try:
                                content.append(str(go_entry[category]))
                            except:
                                try:
                                    content.append(str(go_wise[go][category]))
                                except:
                                    content.append('')
                    content = [str(con) for con in content]
                    ofile.write('{}\n'.format('; '.join(content)))

        self.logger.debug('Wrote per class metrics in full_table.')
        self.plot("", False)
        self.logger.debug("Performed plotting")
