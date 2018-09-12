import os
import sys
import logging
import Bio.AlignIO
from Bio.Align import AlignInfo
from Bio.SubsMat import FreqTable
import numpy as np


class MsaReader():
    """
    Calculates information content
    """
    def set_up_logger(self, info_path):
        """
        Set up the loggers for the module. Also set up filehandlers and streamhandlers (to console).
        """
        # get a logger
        logger = logging.getLogger('msa_reader')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        os.makedirs(info_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(info_path, 'msa_reader.log'))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
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

    def read_msa(self, file_path):
        """
        Read MSA, return MSA as str.
        """
        with open(file_path, 'r') as ifile:
            ifile.readline() # header
            # UPI0009E3695C      ----MSIQHFRVA-LIPFFAAFCLP--VFAHPETLVKVKDAEDKLGARVGYIELDLNSGK
            seqs = {}
            for line in ifile:
                line = line.strip()
                if not line == '':
                    spl = line.split()
                    id = spl[0]
                    seq = spl[1]
                    try:
                        seqs[id] += seq
                    except:
                        seqs[id] = seq
        return seqs

    def count_freqs(self, seqs):
        """
        Count the AA frequencies in given seqs.
        """
        freqs = {}
        for seq in seqs.values():
            for c in seq:
                try:
                    freqs[c] += 1
                except:
                    freqs[c] = 1
        return freqs

    def process_msa(self, file_path):
        """
        Process given MSA (in file_path). Return consensus and information content.
        """
        alignment = Bio.AlignIO.parse(file_path, "clustal")

        alignment = list(alignment)[0]
        self.logger.debug(alignment)
        summary_align = AlignInfo.SummaryInfo(alignment)

        consensus = summary_align.dumb_consensus()
        self.logger.debug('Consensus:\n{}'.format(consensus))

        info_content = []
        for pos in range(len(consensus)):
            info_content.append(summary_align.information_content(start=pos, end=pos+1, e_freq_table=self.freqs))

        return consensus, info_content

    def write_info_content_dump(self, consensus, info_content):
        """
        Dump the consensus sequence and the information content to .tsv.
        """
        mean = np.mean(info_content)
        stdev = np.std(info_content)
        with open('masked_ic_dump.txt', 'a') as ofile:
            ofile.write('>{};{}\n'.format(self.id, 'MSA Information Content'))
            ofile.write('Mean score: {:.3f}, stdev: {:.3f}\n!\n'.format(mean, stdev))
            ofile.write('Pos\tAA\tIC\n')
            seq_chars = ['wt'] + [c for c in consensus]
            info_content = [0.0] + info_content
            self.logger.debug('len(consensus): {} len(info_content): {}'.format(len(consensus), len(info_content)))
            for i in range(len(consensus)):
                ofile.write('{}\t{}\t{}\n'.format(i-1,
                                                  seq_chars[i],
                                                  info_content[i]))

    def write_masked_dataset(self, consensus):
        """
        Writes masked_dataset.txt file containing the consensus sequence.
        :param consensus: consensus sequence
        :return: None
        """
        with open('masked_dataset.txt', 'w') as ofile:
            ofile.write('{};{};{}'.format(self.id, self.go, consensus))
            ofile.flush()

    def __init__(self, save_dir, msa_file_path, pdbid, go):
        # prepare directory
        os.makedirs(save_dir, exist_ok=True)
        os.chdir(save_dir)

        self.logger = self.set_up_logger(save_dir)
        self.logger.info('Called with id {} ({}) for MSA {} to be saved in {}'.format(id, go, msa_file_path, save_dir))
        self.logger.debug('\n\nPrepared directories.\n')

        self.id = pdbid
        self.go = go

        seqs = self.read_msa(msa_file_path)
        self.freqs = FreqTable.FreqTable(self.count_freqs(seqs), FreqTable.COUNT)

        consensus, info_content = self.process_msa(msa_file_path)

        self.write_info_content_dump(consensus, info_content)
        self.write_masked_dataset(consensus)

        self.logger.debug('DONE.')


if __name__ == '__main__':
    msa_file_path = sys.argv[1]
    save_dir = sys.argv[2]
    id = sys.argv[3]
    go = sys.argv[4]

    _ = MsaReader(save_dir, msa_file_path, id, go)