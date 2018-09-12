"""
Pretty plotter for the metrics of the model.
"""
import os
import sys
import logging
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.style.use(json.load(open(os.path.join(sys.path[0], 'style/style.json'), 'r')))

with open(os.path.join(sys.path[0], 'style/colors.json'), 'r') as pickle_file:
    colors = json.load(pickle_file)


class PrettyPlotter():
    def __init__(self, FLAGS, go_info):
        """
        PrettyPLotter needs the flags + go_info to be passed
        :param FLAGS: namespace obj, holds the arguments passed to the main script
        :param go_info: info class obj, hold the information on the nr of classes and the class sizes.
        """
        self.FLAGS = FLAGS
        self.logger = logging.getLogger('{}.plotter'.format(self.FLAGS.modelname))
        self.detailed_plots = (self.FLAGS.detailed_plots == 'True')
        self.go_info = go_info
        self.overall_metrics = None
        self.names  = None
        self.colors = None
        self.width = None
        self.plot_path = os.path.join(os.path.join(self.FLAGS.info_path, 'plots'))
        self.step = 0
        self.early = False
        self.compare = False
        os.makedirs(self.plot_path, exist_ok=True)

        plt.ioff()

    def read_per_class_metrics(self, paths=None, plt_colors=None, names=None):
        """
        Reads the per_class_metrics dumps, into a pandas df for plotting.
        :return:
        """

        if not paths:
            paths = [self.FLAGS.info_path]
        if not names:
            names = [self.FLAGS.modelname]
        if not colors:
            plt_colors = [self.FLAGS.pltcolors.strip().split(',')]

        self.per_class_metrics_list = []

        for path in paths:
            per_class_dir = os.path.join(os.path.join(path, 'metrics'), 'per_class_dumps')
            _per_class_metrics_list = []

            # read the dump for every go that is mentioned in the go_list:
            for go in self.go_info.GOs:
                dump = os.path.join(per_class_dir,
                                    '{}_metrics.npy.npz'.format(go))
                try:
                    _per_class_metrics_list.append(np.load(dump))

                except KeyError or OSError:
                    self.logger.warning('Failed to load per_class_dump for '
                                        '{} with assumed file-path: {}'.format(go, dump))

            self.per_class_metrics_list.append(_per_class_metrics_list)

        self.width = 0.06 * len(self.per_class_metrics_list[0])

        self.logger.info('Finished loading per_class_metrics dumps!')

    def read_over_all_metrics(self, paths=None, names=None, plt_colors=None):
        """
        read the raw predictions and labels dumps
        """
        self.overall_metrics = []
        if not paths:
            paths = [self.FLAGS.info_path]
        if not names:
            names = [self.FLAGS.modelname]
        if not plt_colors:
            plt_colors = self.FLAGS.pltcolors.strip().split(',')

        for path in paths:
            overall_metrics_dump = os.path.join(os.path.join(path, 'metrics'), 'metrics.npy.npz')
            try:
                self.overall_metrics.append(np.load(overall_metrics_dump))
            except KeyError or OSError:
                self.logger.warning('Failed to load overall metrics_dump '
                                    'with assumed file-path: {}'.format(overall_metrics_dump))
        self.names  = names
        self.colors = plt_colors
        while len(self.colors) < len(self.overall_metrics):
            self.colors += self.colors

    def plot_ROC(self):
        """
        Plot the ROC of the model.
        """

        fig, ax = plt.subplots()
        s = 'Model, GOs, AUC\n'
        ax.plot([0, 1.0], [0, 1.0], color=colors['lblue'], lw=2, linestyle="--")

        for model, name, plt_color in zip(self.overall_metrics, self.names, self.colors):

            x = model['fpr']
            y = model['tpr']
            c = colors[plt_color]

            ax.plot(x,
                    y,
                    color=c,
                    lw=2,
                    label=name)
            s += '{}, {}, {:.3f}\n'.format(name, self.go_info.nclasses, model['roc_auc'])

        ax.text(x=0.3,
                y=0.03,
                s=s)

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_title('ROC Curve')

        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        if len(self.overall_metrics) > 1:
            plt.legend()

        plt.savefig(os.path.join(self.plot_path, 'overall_ROC_{}{}.png'.format(self.step, self.early)))
        plt.close(fig)

    def plot_precision_recall(self):
        """
        Plot the precision recall curve of the model.
        """

        fig, ax = plt.subplots()
        s = 'Model, GOs, AUC\n'
        for model, name, color in zip(self.overall_metrics, self.names, self.colors):
            ax.plot(model['recall_arr'], model['precision_arr'], color=colors[color], lw=2, label=name)

            s += '{}, {}, {:.3f}\n'.format(name, self.go_info.nclasses, model['precision_recall_auc'])

        ax.text(x=0.1,
                y=0.03,
                s=s)

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_title('PR Curve')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        if len(self.overall_metrics) > 1:
            plt.legend()

        plt.savefig(os.path.join(self.plot_path, 'overall_precision_recall_{}{}.png'.format(self.step, self.early)))
        plt.close(fig)

    def plot_f1_per_class(self):
        """
        Plot the F1 per class as a barplot.
        """

        self.per_class_metrics_list[0] = sorted(self.per_class_metrics_list[0], key=lambda x: -x['f1'])

        fig, ax = plt.subplots()

        ax.bar(x=list(range(len(self.per_class_metrics_list[0]))),
               height=[x['f1'] for x in self.per_class_metrics_list[0]],
               width=1,
               color=colors['blue'],
               alpha=0.7
              )

        ax.set_ylabel('F1-Score')
        ax.set_xlabel('Class')
        ax.set_title('F1-Score per class')
        plt.savefig(os.path.join(self.plot_path, 'f1_per_class_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_f1_per_class_scatter(self):
        """
        Plot the f1 per class as a scatterplot against n_samples.
        """

        fig, ax = plt.subplots()

        gos = [x['GO'] for x in self.per_class_metrics_list[0]]
        n_samples = [float(self.go_info.key2freq[str(x)]) for x in gos]

        for model, name, color in zip(self.per_class_metrics_list, self.names, self.colors):
            ax.scatter(x=n_samples,
                       y=[x['f1'] for x in model],
                       alpha=0.1,
                       color=colors[color],
                       s=4,
                       label=name
                       )
        if len(self.overall_metrics) > 1:
            plt.legend()
        ax.set_xscale('log')
        ax.set_ylabel('F1 Score')
        ax.set_xlabel('Number of Samples')
        ax.set_title('F1 against N')
        plt.savefig(os.path.join(self.plot_path, 'f1_n_samples_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_roc_auc_per_class(self):
        """
        Plot the ROC AUC per class as a barplot.
        """
        self.per_class_metrics_list[0] = sorted(self.per_class_metrics_list[0], key=lambda x: -float(x['ROC_auc']))
        fig, ax = plt.subplots()

        ax.bar(x=list(range(len(self.per_class_metrics_list[0]))),
               height=[float(x['ROC_auc']) for x in self.per_class_metrics_list[0]],
               width=1,
               color=colors['blue'],
               alpha=0.7
              )
        ax.set_ylabel('ROC AUC')
        ax.set_xlabel('Class')
        ax.set_title('ROC per Class')
        plt.savefig(os.path.join(self.plot_path, 'roc_auc_per_class_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_roc_auc_per_class_scatter(self):
        """
        Plot the ROC AUC per class as a scatterplot against n_samples.
        """
        fig, ax = plt.subplots()

        gos = [x['GO'] for x in self.per_class_metrics_list[0]]
        n_samples = [float(self.go_info.key2freq[str(x)]) for x in gos]

        for model, name, color in zip(self.per_class_metrics_list, self.names, self.colors):
            ax.scatter(x=n_samples,
                       y=[x['ROC_auc'] for x in model],
                       alpha=0.1,
                       color=colors[color],
                       s=4,
                       label=name
                       )
        if len(self.overall_metrics) > 1:
            plt.legend()
        ax.set_xscale('log')
        ax.set_ylabel('ROC AUC')
        ax.set_xlabel('Number of Samples')
        ax.set_title('ROC AUC against N')
        plt.savefig(os.path.join(self.plot_path, 'roc_auc_n_samples_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_precision_recall_auc_per_class(self):
        """
        Plot the F1 per class as a barplot.
        """
        self.per_class_metrics_list[0] = sorted(self.per_class_metrics_list[0],
                                                key=lambda x: -x['precision_recall_auc'])

        fig, ax = plt.subplots()

        ax.bar(x=list(range(len(self.per_class_metrics_list[0]))),
               height=[x['precision_recall_auc'] for x in self.per_class_metrics_list[0]],
               width=1,
               color=colors['blue'],
               alpha=0.7
              )
        ax.set_xlabel('PR AUC')
        ax.set_ylabel('Class')
        ax.set_title('PR AUC per Class')
        plt.savefig(os.path.join(self.plot_path, 'precision_recall'
                                                 '_auc_per_class_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_precision_recall_auc_per_class_scatter(self):
        """
        Plot the precision recall AUC per class as a scatterplot against n_samples.
        """

        fig, ax = plt.subplots()

        gos = [x['GO'] for x in self.per_class_metrics_list[0]]
        n_samples = [float(self.go_info.key2freq[str(x)]) for x in gos]

        for model, name, color in zip(self.per_class_metrics_list, self.names, self.colors):
            ax.scatter(x=n_samples,
                       y=[float(x['precision_recall_auc']) for x in model],
                       alpha=0.1,
                       color=colors[color],
                       s=4,
                       label=name
                       )
        if len(self.overall_metrics) > 1:
            plt.legend()
        ax.set_xscale('log')
        ax.set_title('PR AUC against N')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('PR AUC')
        plt.savefig(os.path.join(self.plot_path, 'precision_recall_auc_'
                                                 'n_samples_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_false_negative_per_class(self):
        """
        Plot the F1 per class as a barplot.
        """
        self.per_class_metrics_list[0] = sorted(self.per_class_metrics_list[0], key=lambda x: -x['fnr'])

        fig, ax = plt.subplots(figsize=(2, 2))

        ax.bar(x=list(range(len(self.per_class_metrics_list[0]))),
               width=1,
               height=[x['fnr'] for x in self.per_class_metrics_list[0]],
               color=colors['blue'],
               alpha=0.7,
               )
        ax.set_ylabel('FNR')
        ax.set_xlabel('Class')
        ax.set_title('FNR per Class')
        plt.savefig(
            os.path.join(self.plot_path, 'fnr_per_class_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_false_negative_scatter(self):
        """
        Plot the precision recall AUC per class as a scatterplot against n_samples.
        """
        fig, ax = plt.subplots()

        gos = [x['GO'] for x in self.per_class_metrics_list[0]]
        n_samples = [float(self.go_info.key2freq[str(x)]) for x in gos]

        for model, name, color in zip(self.per_class_metrics_list, self.names, self.colors):
            ax.scatter(x=n_samples,
                       y=[float(x['fnr']) for x in model],
                       alpha=0.1,
                       color=colors[color],
                       s=4,
                       label=name)
        if len(self.overall_metrics) > 1:
            plt.legend()
        ax.set_xscale('log')
        ax.set_title('FNR against N')
        ax.set_ylabel('FNR')
        ax.set_xlabel('Number of Samples')
        plt.savefig(
            os.path.join(self.plot_path, 'fnr_n_samples_{}{}.png'.format(self.step, self.early)))
        plt.close()

    def plot_all(self, step, early):
        """
        Do all the plots.
        """
        self.logger.debug('Plot all called')

        self.step = step
        if early:
            self.early = '*'
        else:
            self.early = ''
        if self.FLAGS.keep_plots == 'False':
            self.step = ''
            self.early = ''
        try:
            self.read_over_all_metrics()
            self.plot_ROC()
            self.logger.debug('Finished ROC')
            self.plot_precision_recall()
            self.logger.debug('Finished PR')
        except FileNotFoundError:
            self.logger.warning('Unable to plot overall metrics.')

        try:
            self.read_per_class_metrics()
            self.logger.debug('Finished reading per class metrics')

            self.plot_f1_per_class()
            self.logger.debug('Finished plot_f1_per_class')
            self.plot_f1_per_class_scatter()
            self.logger.debug('Finished plot_f1_per_class_scatter')

            self.plot_precision_recall_auc_per_class()
            self.logger.debug('Finished plot_precision_recall_auc_per_class')
            self.plot_precision_recall_auc_per_class_scatter()
            self.logger.debug('Finished plot_precision_recall_auc_per_class_scatter')

            self.plot_roc_auc_per_class()
            self.logger.debug('Finished plot_roc_auc_per_class')
            self.plot_roc_auc_per_class_scatter()
            self.logger.debug('Finished plot_roc_auc_per_class_scatter')
            self.plot_false_negative_per_class()
            self.logger.debug('Finished plot_false_negative_per_class')
            self.plot_false_negative_scatter()
            self.logger.debug('Finished plot_false_negative_scatter')

        except FileNotFoundError:
            self.logger.warning('Unable to plot per class metrics.')

