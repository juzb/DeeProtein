import os
import sys
import logging
import numpy as np
import pandas as pd

"""
Combines masked dumps from replicates of DeeProtein sensitivity analysis by taking the mean at each position for each go 
for each sequence. Assumes paths are impact, impact_1, impact_2, ... writes to combined_impact
First sys.argv is the path in which the impact directories are searched and the combined impact is written. 

"""

pd.options.display.max_rows = 15

class combine:
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

    def run(self):
        paths = [p for p in os.listdir(self.path) if p.startswith(self.prefix) and p[-1].isdigit()]
        
        paths = [os.path.join(os.path.join(self.path, p), dirname) for p in paths]
        self.logger.info('Found the following '
                         'paths to combine:\n{}\nWriting to {}'.format('\n'.join(paths), self.path))

        path_contents = [os.listdir(p) for p in paths]
        min_files = min([len(c) for c in path_contents])
        self.logger.debug('Found a minimum of {} files per directory.'.format(min_files))
        count = 0

        for file in path_contents[0]:
            fuse = True
            for path_content in path_contents:
                if not file in path_content:
                    fuse = False
                    self.logger.debug('Problem with files {}: Not present in all directories.'.format(file))
                    break
            if not fuse:
                continue
            dfs = [pd.read_csv(os.path.join(p, file), sep='\t') for p in paths]
            df = pd.concat(dfs, keys=range(len(dfs)))
            # also save the full dataframe:
            df.to_csv(os.path.join(self.fullpath, file), sep='\t', index_label=['replicate', 'idx'])

            df = df.mean(level=1)

            df['AA'] = dfs[0]['AA']
            df['sec'] = dfs[0]['sec']
            df['dis'] = dfs[0]['dis']

            df.to_csv(os.path.join(self.write_mean, file), sep='\t')
            
            count += 1
            if count % 1000:
                print('{:.1f} % done.'.format(100 * count / len(path_contents[0])))
                
    def __init__(self, path, prefix, dirname):
        self.path = path
        self.prefix = prefix
        self.dirname = dirname
        self.fullpath = os.path.join(self.path, '{}combined_full'.format(prefix))
        self.write_mean = os.path.join(self.path, '{}combined_mean'.format(prefix))
        
        os.makedirs(self.write_mean, exist_ok=True)
        os.makedirs(self.fullpath, exist_ok=True)
        self.logger = self.set_up_logger(self.path, 'combine')

        self.run()
        with open(os.path.join(self.path.replace(self.path.split('/')[-1], ''), 'DONE_COMBINE.txt'), 'w') as _:
            pass
        self.logger.debug('DONE.')


if __name__ == '__main__':
    path = sys.argv[1]
    prefix = sys.argv[2]
    dirname = sys.argv[3]
    _ = combine(path, prefix, dirname)
