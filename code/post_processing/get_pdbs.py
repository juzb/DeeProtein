import os
import sys
import subprocess
import json
import logging


class pdb_getter():
    """
    Prepares masked dataset for evaluation of 3D structure-sensitivity data
    Needs system args:
    1: directory to write the data to
    2: path to GO file
    """
    def set_up_logger(self, info_path):
        """
        Set up the loggers for the module. Also set up filehandlers and streamhandlers (to console).
        """
        # get a logger
        logger = logging.getLogger('get_PDBs')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        os.makedirs(info_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(info_path, 'get_PDBs.log'))
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

    def get_pdb_ids(self):
        """
        Get the pdb IDs.
        """
        if not os.path.exists('pdbs.txt'):
            check = subprocess.check_call(['wget', '-Opdbs.txt', 'https://www.rcsb.org/pdb/rest/getCurrent'])
            if check == 0:
                self.logger.debug('Download of pdb ids successful')
            else:
                self.logger.warning('Download of pdb ids failed.')
        else:
            self.logger.debug('pdbs.txt is already there.')

    def read_pdb_ids(self):
        """
        Read and return the pdb-IDs.
        """
        pdb_ids = []
        with open('pdbs.txt', 'r') as ifile:
            for line in ifile:
                if '<PDB structureId' in line:
                    pdb_ids.append(line.split('"')[1])
        return pdb_ids

    def read_go_file(self, go_file_path):
        """
        Read the GO-file.
        :param go_file_path: `str` path to GO-file.
        :return: `tuple` (list, list) of gos and counts
        """
        gos = []
        counts = []
        with open(go_file_path, "r") as go_fobj:
            for line in go_fobj:
                fields = line.strip().split()
                if fields[1].endswith('.csv'):
                    fields[1] = fields[1].rstrip('.csv')
                gos.append(fields[1])
                counts.append(fields[0])
        return gos, counts

    def get_gos(self, pdb_id):
        """
        Download the GOs for the given pdb ID from rcsb.
        :param pdb_id: `str` PDB ID
        :return: `list` The GOs corresponding to the pdb ID
        """
        gos = []
        check = None
        try:
            check = subprocess.check_call(['wget',
                                           '-Otemp/gos.txt',
                                           'https://www.rcsb.org/pdb/rest/goTerms?structureId={}'.format(pdb_id)])
        except:
            pass
        if check == 0:
            self.logger.debug('Download of gos for pdbid {} successful'.format(pdb_id))
            with open('temp/gos.txt', 'r') as ifile:
                for line in ifile:
                    if '<term id=' in line:  # <term id="GO:0020037" structureId="4HHB" chainId="D">
                        gos.append(line.split('"')[1])
        else:
            self.logger.warning('Download of gos for pdbid {} failed.'.format(pdb_id))
        return gos

    def check_id(self, id):
        """
        Checks, if a PDB entry should be included based on its sequence length
        :param id:  PDB id to check
        :return:    true or false
        """
        self.get_fasta_file(id)
        seq = self.read_fasta(id)
        return (150 < len(seq) <1000)

    def get_pdb_file(self, id):
        """
        Download the PDB-file w/ the specified ID from rcsb.
        :param id: 'str' the PDB ID to download.
        :return: `bool` whether download was successful
        """
        if os.path.exists('pdbs/{}.pdb'.format(id)):
            self.logger.debug('pdb   file {} already downloaded'.format(id))
            return
        check = None
        try:
            check = subprocess.check_call(['wget',
                                           '-Opdbs/{}.pdb'.format(id),
                                           'https://files.rcsb.org/download/{}.pdb'.format(id)])
        except:
            pass
        if check == 0:
            self.logger.debug('Download of pdb file for pdbid {} successful'.format(id))
            return True
        else:
            self.logger.warning('Download of pdb file for pdbid {} failed.'.format(id))
            return False

    def get_fasta_file(self, id):
        """
        Download FASTA-file for specified ID.
        :param id: 'str' the ID to download the fasta for.
        :return:
        """
        if os.path.exists('fastas/{}.fasta'.format(id)):
            self.logger.debug('fasta file {} already downloaded'.format(id))
            return
        check = None
        try:
            check = subprocess.check_call(['wget',
                                           '-Ofastas/{}.fasta'.format(id),
                                           'https://www.rcsb.org/pdb/download'
                                           '/downloadFastaFiles.do?structureIdList='
                                           '{}&compressionType=uncompressed'.format(id)])
        except:
            pass
        if check == 0:
            self.logger.debug('Download of fasta file for pdbid {} successful'.format(id))
        else:
            self.logger.warning('Download of fasta file for pdbid {} failed.'.format(id))

    def read_fasta(self, id):
        """
        Reads teh fasta and parses to string.
        :param id: 'str' FASTA to read.
        :return: sequence as str
        """
        with open('fastas/{}.fasta'.format(id)) as ifile:
            ifile.readline()
            out = ''
            for line in ifile:
                line.strip()
                if line.startswith('>'):
                    break
                out += line
        return out

    def write_masked_dataset(self):
        """
        Writes the masked_dataset.txt file that is read for the sensitivity analysis
        :return: None
        """
        with open('masked_dataset.txt', 'w') as ofile:
            for go in self.go2id.keys():
                pdb_id = self.go2id[go]
                seq = self.read_fasta(pdb_id)
                ofile.write('{};{};{}\n'.format(pdb_id, go, seq))

    def save_all(self):
        """
        Save all conversions.
        """
        with open('go2id.json', 'w') as ofile:
            json.dump(self.go2id, ofile)
        with open('go_save.txt', 'w') as ofile:
            ofile.write(','.join(self.gos_to_search))

        # prepare txt for masked dataset
        self.write_masked_dataset()
        self.logger.debug('\n\nSaved everything.\n')

    def __init__(self, save_dir, go_file_path):

        # prepare directory

        save_interval = 100
        os.makedirs(save_dir, exist_ok=True)
        os.chdir(save_dir)

        self.logger = self.set_up_logger(save_dir)

        self.logger.debug('\n\nPrepared directories.\n')
        # get pdb ids
        self.get_pdb_ids()

        self.logger.debug('\n\nGot pdb id list.\n')
        pdb_ids = self.read_pdb_ids()

        self.logger.debug('\n\nGot {} pdb ids.\n'.format(len(pdb_ids)))

        # read gofile

        self.gos_to_search, counts = self.read_go_file(go_file_path)
        self.logger.debug('\n\nRead GO file.\n')

        # get gos
        temp_dir = os.path.join(save_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # get ids for interesting gos
        fastas_dir = os.path.join(save_dir, 'fastas')
        os.makedirs(fastas_dir, exist_ok=True)

        pdbs_dir = os.path.join(save_dir, 'pdbs')
        os.makedirs(pdbs_dir, exist_ok=True)

        self.go2id = {}
        count = 0
        self.logger.debug('Saving every {} PDB ids.'.format(save_interval))
        for pdb_id in pdb_ids:
            if count % save_interval == 0:
                self.save_all()
                self.logger.debug('Pdb id No. {}'.format(count))

            gos = self.get_gos(pdb_id)
            for go in gos:
                if go in self.gos_to_search:
                    if pdb_id not in self.go2id.values():
                        if self.check_id(pdb_id):
                            if self.get_pdb_file(pdb_id):
                                self.go2id[go] = pdb_id
                                self.gos_to_search.remove(go)
                                if len(gos) == 0:
                                    break
            count += 1

        # final save
        self.save_all()
        self.logger.debug('DONE.')


if __name__ == '__main__':
    save_dir = sys.argv[1]
    go_file_path = sys.argv[2]
    _ = pdb_getter(save_dir, go_file_path)