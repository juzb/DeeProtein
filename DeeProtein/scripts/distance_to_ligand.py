import os
import sys
import logging
import json
import scipy.stats as stats
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

style_path = 'path/to/style/directory'


plt.style.use(json.load(open(os.path.join(style_path, 'style.json'), 'r')))
with open(os.path.join(style_path, 'colors.json'), 'r') as pickle_file:
    colors = json.load(pickle_file)

"""
First system argument is the path to the directory in which the data is:
binding activities.csv with the following colummns PDB;Chain;GO;Ligand;Organism;Name;Comment
a directory called 'mean_ligand_binding', in which the masked dumps are

A new plots directory will be made, containing the outputs.

Specify a style-path above. This has to contain a colors.json, style.json

"""
class evaluate():
    """
    Analyses sensitivity data based on distance to a ligand
    """
    def set_up_logger(self, info_path, name):
        """
        Set up the loggers for the module. Also set up filehandlers and streamhandlers (to console).
        """
        # get a logger
        logger = logging.getLogger('{}'.format(name))
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        os.makedirs(info_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(info_path, 'evaluate.log'))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info('\n\n\nNew call:\n=========\n\n\n')
        return logger


    def read_masked_dump(self, filepath):
        """
            Reads the information from one file, returns it in accessible format.
            :param file_path:   Path to the file
            :param filename:    Name of the file
            :return:            the data as a dict: keys are headers of the file, values are lists corresponding to the lines,
                                the pdb identifier, the chain identifier, the mask-width (always 1 in this study)
        """

        if not os.path.isfile(filepath):
            return False, filepath, False, False

        count = 0
        with open(filepath, 'r') as ifile:
            keys = ifile.readline().strip().split('\t')

            data = {}
            for key in keys:
                data[key.strip()] = []
            for line in ifile:
                count += 1
                if line.startswith('>'):
                    break
                for entry, key in zip(line.strip().split('\t'), keys):
                    try:
                        data[key.strip()].append(float(entry))
                    except:
                        data[key.strip()].append(entry)

        if count < 3:
            return False

        return data

    def read_todos(self, path):
        """
        Reads CSV file specified in path, parses to dict
        :param path: file to read
        :return: dict containing the todo information as dicts for each line
        """

        out = {}
        with open(path, 'r') as ifile:
            head = ifile.readline().strip().split(';')
            for line in ifile:
                out[line.split(';')[0]] = dict(zip(
                    head,
                    line.strip().split(';')
                ))

        return out


    def separate_values(self, data, category):
        """
        TODO docstr
        :param data:        The data to separate as dict with 'category' and 'd-lig' as keys
        :param category:    category to compare for close vs. distant residues
        :return:            the values of 'category' that are close/distant as two lists
        """
        close   = []
        distant = []

        for sen, d in zip(data[category], data['d-lig']):
            if d < self.radius:
                close.append(sen)
            else:
                distant.append(sen)

        return close, distant


    def plot_and_test(self, c, d, t):
        """
        Plots and performs statistic
        :param c: list of close values
        :param d: list of distant values
        :param t: todos, i.e. information in a dict with the following keys: 'PDB', 'Chain', 'GO', 'Name', 'Ligand'
        :return: t-statistic, p-value determined by one-sided Welch's t-test
        """
        statistic, p = stats.ttest_ind(c, d, equal_var=False, nan_policy='omit') # welchs t-test, two tailored p-value
        p /= 2.0
        if statistic > 0:
            p = 1-p


        fig, ax = plt.subplots(figsize=[1.0, 1.5])

        parts = ax.violinplot([c, d],
                              showmeans=True,
                              showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor(colors['blue'])
        for pn in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            parts[pn].set_edgecolor(colors['blue'])

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Close\n{}'.format(len(c)), 'Distant\n{}'.format(len(d))])

        plt.savefig(os.path.join(self.plot_path, '{}_{}_{}_{}-{}.pdf'.format(t['PDB'],
                                                                             t['Chain'],
                                                                             t['GO'].replace(':', '-'),
                                                                             t['Name'],
                                                                             t['Ligand']).replace('/', '-')))

        plt.close()


        return statistic, p


    def plot_dist(self, data):
        """
        Plot the distribution of P-values.
        :param data:        list of datapoints to plot
        :return:
        """
        data = sorted(data, key=lambda x: -x)
        fig, ax = plt.subplots()

        data = np.log(data)
        borders = ax.hist(data, range=(np.log(0.0001), 0.0))
        self.logger.info('Histogram borders: {}'.format(', '.join([str(e) for e in np.exp(borders[1])])))
        locs, labels = plt.xticks()
        ax.set_xlim([np.log(0.0001), 0.0])
        labels = np.exp([l for l in locs])
        labels = ['{:.1e}'.format(l) for l in labels]
        plt.xticks(locs, labels, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'distribution.pdf'))
        plt.close()

    def run(self):
        """
        Run the analysis and plot results.
        Also logs PDB ID, Chain ID, p-value, t-value and ligand name in that order for each example
        """
        ipath = os.path.join(self.path, 'mean_ligand_binding')

        files = [f for f in os.listdir(ipath) if f.endswith('.txt')]
        files = sorted(files)
        self.logger.debug('Files:\n{}'.format('\n'.join(files)))
        todos = self.read_todos(os.path.join(self.path, 'binding_activities.csv'))


        count = 0
        ps = []
        stat_values = []
        with open(os.path.join(self.path, 'headers.txt'), 'w') as head_file:

            for file in files:
                _, pdbid, chain, _ = file.split('_')

                if pdbid in todos:
                    t = todos[pdbid]

                    data = self.read_masked_dump(os.path.join(ipath, file))

                    c, d = self.separate_values(data, t['GO'])

                    stat, p = self.plot_and_test(c, d, t)
                    ps.append(p)
                    stat_values.append(p)
                    self.logger.debug('{} {} - {:.1e}, {:.2f} \t-> {}'.format(t['PDB'], t['Chain'], p, stat, t['Ligand']))

                    t['p'] = p
                    t['stat'] = stat
                    t['nc'] = len(c)
                    t['nd'] = len(d)
                    #letter  = alphabet[count % len(alphabet)]
                    org     = t['Organism']
                    name    = t['Name']
                    lig     = t['Ligand']
                    go      = t['GO']

                    head_file.write('{} {} - {}\n{} {}, {}, p = {:.3f}, t = {:.2f}\n\n\n'.format(
                        org, name, lig, pdbid, chain, go, p, stat
                    ))
                    count += 1
                else:
                    print('{} {} not in binding_activities.csv.'.format(pdbid, chain))

        self.logger.debug('Successful for {} proteins.'.format(len(ps)))
        self.logger.debug('p < 0.05 for {} proteins.'.format(len([p for p in ps if p < 0.05 and p > 0.0])))

        self.plot_dist(ps)

    def __init__(self, path):
        """
        :param path: the path to the directory in which the data is:
                        binding activities.csv with the following colummns PDB;Chain;GO;Ligand;Organism;Name;Comment
                        a directory called 'mean_ligand_binding', in which the masked dumps are

                        A new plots directory will be made, containing the outputs.
        """

        self.path = path
        self.plot_path = os.path.join(path, 'plots')
        os.makedirs(self.plot_path, exist_ok=True)
        self.logger = self.set_up_logger(path, 'd-lig')
        self.radius = 20
        self.percentage = 5
        self.run()
        self.logger.debug('DONE.')


if __name__ == '__main__':
    path = sys.argv[1]
    _ = evaluate(path)
