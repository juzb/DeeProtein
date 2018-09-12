import os
import sys
import logging
import numpy as np


"""
Combines masked dumps from replicates of DeeProtein sensitivity analysis by taking the mean at each position for each go 
for each sequence. Assumes paths are impact, impact_1, impact_2, ... writes to combined_impact
First sys.argv is the path in which the impact directories are searched and the combined impact is written. 

"""
class combine():
    """
    Handles all functionality.
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
        fh = logging.FileHandler(os.path.join(info_path, 'combine.log'))
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

    def process_line(self, tmp):
        """
        Combines a single line from all files into one line containing the mean.
        """
        outline_mean = []
        lines = []
        for line in tmp:
            lines.append(line.strip().split())

        for pos in range(len(lines[0])):
            if pos > 1:
                number = True
                try:
                    float(lines[0][pos])
                except:
                    number = False
            else:
                number = False # to prevent taking the variance of the position
            if number:
                values = [float(lines[f_idx][pos]) for f_idx in range(len(lines))]
                outline_mean.append(float(np.nanmean(values)))
            else:
                outline_mean.append(lines[0][pos])

        outline_mean = '{}\n'.format('\t'.join([str(o) for o in outline_mean]))

        return outline_mean

    def run(self):
        """
        Finds all paths that will be combined, finds all files that are present in all directories, and not already written
        in the target directory. Reads all files, calculates mean and writes single new file.
        """
        paths = [p for p in os.listdir(self.path) if p.startswith('impact')]
        paths = [os.path.join(os.path.join(self.path, p), 'aa_resolution') for p in paths]
        self.logger.info('Found the following '
                         'paths to combine:\n{}\nWriting to {}'.format('\n'.join(paths), self.outpath))

        already_written = [f for f in os.listdir(self.write_mean) if f.endswith('.txt') and f.startswith('masked_')]

        path_contents = [os.listdir(p) for p in paths]
        min_files = min([len(c) for c in path_contents])
        self.logger.debug('Found a minimum of {} files per directory.'.format(min_files))
        count = 0

        for file in path_contents[0]:
            if file in already_written:
                self.logger.debug('Problem with files {}: already written.'.format(file))
                continue

            fuse = True
            for path_content in path_contents:
                if not file in path_content:
                    fuse = False
                    self.logger.debug('Problem with files {}: Not present in all directories.'.format(file))
                    break
            if not fuse:
                continue

            # file present in all paths:
            ifiles = [open(os.path.join(p, file), 'r') for p in paths]
            first_lines = []
            same = True
            for f in ifiles:
                first_lines.append(f.readline())
            first_line = first_lines[0]

            for fl in first_lines:
                if fl != first_line:
                    self.logger.debug('Problem with files {}:\nFirst line\n{}\n{}'.format(file, first_line, fl))
                    same = False
                    break
            if not same:
                continue

            with open(os.path.join(self.write_mean, file), 'w') as ofile_mean:
                ofile_mean.write(first_line)
                lines = [f.readlines() for f in ifiles]

                for l_idx in range(len(lines[0])):
                    mean = self.process_line([lines[f_idx][l_idx] for f_idx in range(len(lines))])
                    ofile_mean.write(mean)
                count += 1

    def __init__(self, path):
        self.path = path
        self.outpath = os.path.join(path, 'combined_impact')
        self.write_mean = os.path.join(self.outpath, 'aa_resolution')
        os.makedirs(self.write_mean, exist_ok=True)
        self.logger = self.set_up_logger(self.outpath, 'combine')

        self.run()
        with open(os.path.join(self.path.replace(self.path.split('/')[-1], ''), 'DONE_COMBINE.txt'), 'w') as _:
            pass
        self.logger.debug('DONE.')


if __name__ == '__main__':
    path = sys.argv[1]
    _ = combine(path)
