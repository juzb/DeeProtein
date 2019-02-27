import re
import os
import glob
import sys
import subprocess as sb
from goatools.obo_parser import GODag
from collections import OrderedDict


class DataPreprocessor:
    def __init__(self, save_dir, read_dir=None, func_only=False, godagfile=''):
        self.filter_AA = True
        self.filter_length = True
        self.minlength = 0
        self.maxlength = 1000

        if not func_only:
            self.GODag = GODag(godagfile,
                               optional_attrs=['relationship'])

            # set save dir and ensure exists:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir

            if not read_dir:
                self.read_dir = save_dir
            else:
                self.read_dir = read_dir

            self.logfile = open(os.path.join(self.save_dir, 'logfile.txt'), 'w')
            self.max_write = None
            self.logfile.write('Initialised.\n')
            self.logfile.flush()

    def uniprot_csv_parser(self, in_fobj):
        """Parser for the uniprot.csv.

        The uniprot csv file is in the following syntax:
        name;rec_name;Pfam;protein_existance;seq;F_GO;P_GO;C_GO;EC
        0   ;1       ;2   ;3                ;4  ;5   ;6   ;7   ;8
        or
        name;seq;F_GO;P_GO;C_GO;EC
        0   ;1  ;2   ;3   ;4   ;5
        Args:
          in_fobj: `fileObject`

        Yields:
          name: 'str' the name.
          seq: 'str' the sequence.
          GO: `list` holding the GO labels of a sequence.
          EC: 'str' holding the EC label.
          structure_str: 'str' defninig the secondary structure of the sequence.
        """
        for line in in_fobj:
            fields = line.strip().split(';')

            name = fields[0]
            # str to list
            seq = fields[1]
            fGO = fields[2].split(',')
            pGO = fields[3].split(',')
            cGO = fields[4].split(',')
            EC = fields[5].split(',')
            yield name, seq, fGO, pGO, cGO, EC

    def uniprot_to_csv_on_disk(self, uniprot_filepath):
        """Convert the raw uniprot download into a csv file.

        Converts the raw uniprot.txt download (specified in the class_attributes) into a .csv file with the
        following syntax:

        name;rec_name;Pfam;protein_existance;seq;F_GO;P_GO;C_GO;EC

        After each entry the information is written, to avoid a memory explosion.
        """
        uniprot_dict = {}
        uniprot_csv_path = os.path.join(self.save_dir, 'filtered_uniprot.csv')
        out_csv = open(uniprot_csv_path, 'a')
        line_number = 0
        with open(uniprot_filepath, "r") as in_fobj:
            curr_prot_id = ''
            curr_F_GOs = []
            curr_P_GOs = []
            curr_C_GOs = []
            curr_ECs = []
            seq = False
            for line in in_fobj:
                line_number += 1
                if line_number % 1000000 == 0:
                    self.logfile.write('{} Million lines\n'.format(line_number / 1000000))
                    self.logfile.flush()
                fields = line.strip().split()
                flag = fields[0]
                if flag == 'ID' and len(fields) >= 2:
                    curr_prot_id = fields[1]
                    uniprot_dict[curr_prot_id] = {}
                elif flag == 'DE' and len(fields) >= 2:
                    rec_name = re.search(r'(?<=Full=)(.+?)[;\s]', line)
                    ec_nr = re.search(r'(?<=EC=)([0-9.-]*?)[;\s]', line)
                    if ec_nr:
                        curr_ECs.append(ec_nr.group(1))
                    elif rec_name:
                        uniprot_dict[curr_prot_id]['rec_name'] = rec_name.group(1)
                elif flag == 'DR' and len(fields) >= 2:
                    '''
                    abfrage fuer GOS und PFAM
                    '''
                    # ask for GO first:
                    dr_fields = [ref.rstrip('.;') for ref in fields[1:]]
                    if dr_fields[0] == 'GO' and dr_fields[2].startswith('F'):
                        curr_F_GOs.append(dr_fields[1].strip(';'))
                    elif dr_fields[0] == 'GO' and dr_fields[2].startswith('P'):
                        curr_P_GOs.append(dr_fields[1].strip(';'))
                    elif dr_fields[0] == 'GO' and dr_fields[2].startswith('C'):
                        curr_C_GOs.append(dr_fields[1].strip(';'))

                    elif dr_fields[0] == 'Pfam':
                        uniprot_dict[curr_prot_id]['Pfam'] = dr_fields[2:]
                    else:
                        pass
                elif flag == 'CC' and len(fields) >= 2:
                    '''
                    may content sequence caution warning
                    '''
                    pass

                elif flag == 'PE' and len(fields) >= 2:
                    protein_existance = fields[1]
                    uniprot_dict[curr_prot_id]['protein_existance'] = protein_existance

                elif flag == 'SQ' and len(fields) >= 2:
                    seq = True
                    uniprot_dict[curr_prot_id]['seq'] = ''
                elif seq:
                    if flag == '//':
                        uniprot_dict[curr_prot_id]['F_GO'] = self._full_annotation(curr_F_GOs)
                        uniprot_dict[curr_prot_id]['P_GO'] = self._full_annotation(curr_P_GOs)
                        uniprot_dict[curr_prot_id]['C_GO'] = self._full_annotation(curr_C_GOs)
                        uniprot_dict[curr_prot_id]['EC'] = curr_ECs

                        # write the entry to file
                        csv_entry = "{};".format(curr_prot_id)
                        for key in ['seq']:  # 'rec_name', 'Pfam', 'protein_existance',
                            try:
                                csv_entry += "{};".format(uniprot_dict[curr_prot_id][key])
                            except KeyError:
                                csv_entry += ";"

                        # write the GOs seperated by comma:
                        for category in ['F_GO']:  # ['F_GO', 'P_GO', 'C_GO']:
                            # those are already strings, so we can simply add them
                            csv_entry += '{};'.format(uniprot_dict[curr_prot_id][category])

                        # close entry
                        csv_entry += "\n"

                        if self._valid_seq(uniprot_dict[curr_prot_id]['seq']):
                            if len(uniprot_dict[curr_prot_id]['F_GO']) > 0:
                                out_csv.write(csv_entry)
                        # <curr_prot_id>;<rec_name>;<Pfam>;<protein_existance>;<seq>;<F_GO1, F_GO2, ...>;<P_GO>;<C_GO>;<EC>

                        # reset collectors:
                        curr_prot_id = ''
                        seq = False
                        curr_F_GOs = []
                        curr_C_GOs = []
                        curr_P_GOs = []
                        curr_ECs = []

                        uniprot_dict = {}
                    else:
                        uniprot_dict[curr_prot_id]['seq'] += ''.join(fields)
                else:
                    pass
        out_csv.close()

    def separate_classes_by_GO(self, uniprot_csv):
        """Seperates the whole uniprot.csv into GO-specific .csv-files.

        First generates a GO to ID dict, then split the uniprot.csv into GO-term specific files.

        Args:
          jobnr: 'str', a jobnumber if this funciton is used in a jobarray to handle the whole uniprot.csv (optional)
        """
        # generate extra folder in savedir to store the single files in
        csv_by_GO_paths = OrderedDict([('fGO', os.path.join(self.save_dir, 'csv_by_fGO')),
                                       ('pGO', os.path.join(self.save_dir, 'csv_by_pGO')),
                                       ('cGO', os.path.join(self.save_dir, 'csv_by_cGO'))])

        for _, item in csv_by_GO_paths.items():
            if not os.path.exists(item):
                os.mkdir(item)

        # get a GO-dict to store the population of all GO-terms we have. Dict is flat and not lvl wise (as wen do not
        # need to reconstruct the DAG.
        self.GO_population_dict = {}

        omitted_GOs = []

        # got through the uniprot csv once and set up a dict of all GO-terms. Pass an ID
        lines_read = 0
        with open(uniprot_csv, "r") as in_fobj:
            in_fobj.readline()
            for name, seq, fGO, pGO, cGO, EC_nrs in self.uniprot_csv_parser(in_fobj):
                lines_read += 1
                if lines_read % 1000000 == 0:
                    print('Read {}m lines.\n'.format(lines_read / 1000000))
                # iterate through the whole GO annotation and complete it by checking for parents in the DAG
                if self._valid_seq(seq):
                    full_go = set()
                    # if __name__ == '__main__':
                    for GOcat in [(fGO, 'fGO')]:
                        for go in GOcat[0]:
                            # determine if a node has parents and retrieve the set of parent-nodes
                            try:
                                full_go.update(self.GODag[go].get_all_parents())
                                full_go.add(go)
                            except KeyError:
                                if go:
                                    self.logfile.write(go)
                                # this means the term might be obsolete as its not in the DAG. store it
                                omitted_GOs.append((name, go))
                                pass

                        # sort the full annotation by levels:
                        full_go_by_level = {}
                        levels = set([self.GODag[go].level for go in list(full_go)])
                        for lvl in sorted(list(levels)):
                            full_go_by_level[lvl] = [go for go in list(full_go) if self.GODag[go].level == lvl]

                        for lvl, go_terms in full_go_by_level.items():
                            # construct the line to be written:
                            line = [name, seq]

                            full_go_list = list(full_go)

                            if GOcat[1] == 'fGO':
                                line.append(','.join(full_go_list))
                            else:
                                line.append(','.join(fGO))

                            if GOcat[1] == 'pGO':
                                line.append(','.join(full_go_list))
                            else:
                                line.append(','.join(pGO))

                            if GOcat[1] == 'cGO':
                                line.append(','.join(full_go_list))
                            else:
                                line.append(','.join(cGO))

                            line.append(','.join(EC_nrs))

                            line = ';'.join(line)

                            line += '\n'
                            # open the corresponding csvs and add the line from the uniprot csv.
                            for go_term in go_terms:
                                # update the counters for the population dict
                                try:
                                    self.GO_population_dict[go_term] += 1
                                except KeyError:
                                    self.GO_population_dict[go_term] = 1

                                with open(os.path.join(csv_by_GO_paths[GOcat[1]],
                                                       '{}.csv'.format(go_term)), "a") as go_csv:
                                    go_csv.write(line)
        print('Total lines read: {}'.format(lines_read))

    def _full_annotation(self, GO_terms):
        """Takes a list of GO_terms and expands them to full annotation.

        Completes the subdag defined by a list of GO-terms by walking up the GOdag.

        Args:
          GO_terms: `list` the GO_terms to expand.
        Returns:
          A fully annotated list of GO terms including all nodes passed on the way to root.
        """
        full_go = set()

        omitted_GOs = []

        for go in GO_terms:
            # determine if a node has parents and retrieve the set of parent-nodes
            try:
                full_go.update(self.GODag[go].get_all_parents())
                full_go.add(go)
            except KeyError:
                # this means the term might be obsolete as its not in the DAG. store it
                omitted_GOs.append(go)
                pass

        full_go = ','.join(list(full_go))

        return full_go

    def _valid_seq(self, seq):
        """Check a sequence for forbidden AAs and minlength.

        A helper function for filter_and_write().

        Args:
          seq: 'str' the sequence to check.

        Returns:
          A `bool` whether the sequence is valid (True) or not.
        """
        try:
            if re.search(r'[A-Z]+', seq).span() != (0, len(seq)):
                return False
        except:
            return False
        if self.filter_AA and self.filter_length:
            forbidden_AAs = re.search(r'[BXZOUJ]', seq)
            if int(self.minlength) <= len(seq) <= int(self.maxlength) and not forbidden_AAs:
                return True
        elif self.filter_AA and not self.filter_length:
            forbidden_AAs = re.search(r'[BXZOUJ]', seq)
            if not forbidden_AAs:
                return True
        elif not self.filter_AA and self.filter_length:
            if int(self.minlength) <= len(seq) <= int(self.maxlength):
                return True
        else:
            return False

    def remove_duplicates(self):
        """Generates duplicate free csvs for each GO-term, duplicates between different GO-term files may still exist.
        For every file: reads every line, adds the sequence to a set of all previous sequences, if that set does not
        grow, its a duplicate, else the line is written to the new file"""

        function_csvs = glob.glob(os.path.join(
            os.path.join(self.save_dir, 'csv_by_fGO'),
            '*.csv')
        )
        self.logfile.write('function csvs:\n{}\n'.format(function_csvs))

        if not os.path.exists(os.path.join(self.save_dir, 'csv_by_fGO_duplicate_free')):
            os.mkdir(os.path.join(self.save_dir, 'csv_by_fGO_duplicate_free'))

        for csv_file in function_csvs:
            self.logfile.write(csv_file)
            self.logfile.write('\n')
            self.logfile.flush()
            num_duplicates = 0
            num_seqs = 0
            with open(csv_file, 'r') as current_file:
                with open(csv_file[::-1].replace('csv_by_fGO'[::-1],
                                                 'csv_by_fGO_duplicate_free'[::-1])[::-1],
                          'w') as csv_file_duplicate_free:

                    # make a set with all sequences from the current file
                    all_seqs = set()

                    for line in current_file:
                        old_len = len(all_seqs)
                        all_seqs.add(line.split(';')[1])
                        if len(all_seqs) == old_len + 1:
                            csv_file_duplicate_free.write(line)
                        else:
                            num_duplicates += 1
                        num_seqs += 1
            self.logfile.write('Found {} duplicates in a total of {} sequences ({} %).\n\n'.format(
                num_duplicates, num_seqs, (100 * num_duplicates / num_seqs)
            ))

    def generate_dataset_by_GO_list(self, GO_file, GO_cat='function'):
        """Generate the train and validsets from a given GO-File.
        Generate a dataset from the passed list of GO_terms. Sequences are included only once, even though they might
        be present in multiple GO-term files. The annotated GO-terms are filtered for the passed GO-terms and those
        not included in the passed list are omitted.

        In parallel a valid set is generated by randomly choosing 5 samples per class.

        Args:
        GO_file: A .txt file containing the GOs to be considered. One GO per line.
        """
        self.logfile.write('Generating dataset from GO_list\n')
        assert GO_cat in ['function', 'pathway', 'location']

        csv_by_GO_paths = OrderedDict([('function', os.path.join(self.read_dir, 'csv_by_fGO_duplicate_free')),
                                       ('pathway', os.path.join(self.read_dir, 'csv_by_pGO_duplicate_free')),
                                       ('location', os.path.join(self.read_dir, 'csv_by_cGO_duplicate_free'))])

        # First shuffle all of the per GO csvs:
        for _, path in csv_by_GO_paths.items():
            per_class_csvs = glob.glob(os.path.join(path, '*.csv'))
            for fpath in per_class_csvs:
                if not os.path.exists('{}.shuffled'.format(fpath)):
                    sb.run(['shuf', fpath, '-o', '{}.shuffled'.format(fpath)])

        # read in the GO_file:
        GOs_to_include = []
        with open(GO_file, "r") as infileobj:
            for line in infileobj:
                curr_GO = line.strip().split()[1]
                GOs_to_include.append(curr_GO)

        self.logfile.write('Read GO_file. Found {} GO-Terms.\n'.format(len(GOs_to_include)))

        # now iterate over the GOs and write the corresponding sequences into a three datasets
        # we take the 10 % for the test set, 20 % for the validation set and make sure that
        # they are NOT present in the training set (70 %).
        if not os.path.exists(os.path.join(self.save_dir, 'datasets')):
            os.mkdir(os.path.join(self.save_dir, 'datasets'))

        train_path = os.path.join(self.save_dir, 'datasets/{}_{}_TRAIN.csv'.format(GO_cat, len(GOs_to_include)))
        valid_path = os.path.join(self.save_dir, 'datasets/{}_{}_VALID.csv'.format(GO_cat, len(GOs_to_include)))
        test_path = os.path.join(self.save_dir, 'datasets/{}_{}_TEST.csv'.format(GO_cat, len(GOs_to_include)))
        counter = 0

        with open(train_path, 'w') as train_fobj:
            with open(valid_path, 'w') as valid_fobj:
                with open(test_path, 'w') as test_fobj:
                    test_seqs = set()
                    valid_seqs = set()
                    train_seqs = set()

                    self.logfile.write('Number of sequences in the different sets:\nGO-term:\tTest:\tTrain:\tValid:\n')
                    for GO in GOs_to_include:
                        with open(os.path.join(csv_by_GO_paths[GO_cat], '{}.shuffled'.format(GO)), 'r') as in_fobj:
                            # get the number of lines:
                            num_lines = sum(1 for line in in_fobj)

                            if self.max_write:
                                num_lines = max([num_lines, self.max_write])
                            # get back to the beginning of the file
                            in_fobj.seek(0)
                            old_len_test_seqs = len(test_seqs)
                            old_len_valid_seqs = len(valid_seqs)
                            old_len_train_seqs = len(train_seqs)
                            # parse the go.csv
                            for name, seq, fGO, pGO, cGO, EC_nrs in self.uniprot_csv_parser(in_fobj):

                                # put only the most relevant information in the dataset file: name, seq, GOs that matter
                                GOs = {GO.strip('.csv')}
                                if GO_cat == 'function':
                                    for el in fGO:
                                        if '{}.csv'.format(el) in GOs_to_include:
                                            GOs.add(el)
                                elif GO_cat == 'pathway':
                                    for el in pGO:
                                        if '{}.csv'.format(el) in GOs_to_include:
                                            GOs.add(el)
                                else:
                                    for el in cGO:
                                        if '{}.csv'.format(el) in GOs_to_include:
                                            GOs.add(el)

                                line = ';'.join([name, seq, ','.join(GOs)]) + '\n'

                                if len(test_seqs) <= int(0.1 * num_lines):
                                    if seq not in test_seqs:
                                        # write to test
                                        test_fobj.write(line)
                                        # store the sequence to compare all the training seqs to this:
                                        test_seqs.add(seq)
                                        counter += 1
                                elif len(valid_seqs) <= int(0.2 * num_lines):
                                    if seq not in valid_seqs and seq not in test_seqs:
                                        # write to valid
                                        valid_fobj.write(line)
                                        # store the sequence to compare all the training seqs to this:
                                        valid_seqs.add(seq)
                                        counter += 1
                                else:
                                    if seq not in train_seqs and seq not in valid_seqs and seq not in test_seqs:
                                        # write to valid
                                        train_fobj.write(line)
                                        # store the sequence to compare all the training seqs to this:
                                        train_seqs.add(seq)
                                        counter += 1

                        self.logfile.write('{}:\t{}\t{}\t{}\n'.format(GO,
                                                                      len(test_seqs) - old_len_test_seqs,
                                                                      len(valid_seqs) - old_len_valid_seqs,
                                                                      len(train_seqs) - old_len_train_seqs))
            self.logfile.write('\nFinished writing the sets consisting of a total of {} sequences.\n\n'.format(counter))
            self.logfile.flush()

            del test_seqs, valid_seqs, train_seqs

            # shuffle the new files
            sb.run(['shuf', train_path, '-o', '{}.shuffled'.format(train_path)])
            sb.run(['shuf', valid_path, '-o', '{}.shuffled'.format(valid_path)])
            sb.run(['shuf', test_path, '-o', '{}.shuffled'.format(test_path)])
            self.logfile.write('This Shuffling step is unreliable, better shuffle the files manually: shuf in > out\n')
            self.logfile.write('Shuffled the datasets.\n')


if __name__ == '__main__':
    save_dir = sys.argv[1]
    up_file = sys.argv[2]
    godagfile = sys.argv[3]

    datasetgen = DataPreprocessor(save_dir, None, False, godagfile)
    datasetgen.uniprot_to_csv_on_disk(up_file)
