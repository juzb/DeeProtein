import os
import sys
from shutil import copyfile
import logging
import json
import scipy.stats as stats
import numpy as np
import Bio.PDB as pdb
from goatools.obo_parser import GODag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[0].replace('post_processing/evaluate_masked_dump.py', 'helpers'))
from helpers_cpu import *

np.warnings.simplefilter('ignore', np.RankWarning)
plt.style.use(json.load(open(os.path.join(sys.path[0], '../style/style.json'), 'r')))

with open(os.path.join(sys.path[0], '../style/colors.json'), 'r') as pickle_file:
    colors = json.load(pickle_file)


class evaluate():
    """
    Read masked dump files that contain the sensitvity values. Calculates sphere variance, compares with information
    content.
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

    def count_dms(self):
        """
        Prepares biological activity data in the same format as the senstivity
        """
        os.makedirs(os.path.join(self.path, 'aa_resolution_dms'), exist_ok=True)
        files = os.listdir(self.dms_path)
        for file in files:
            pdbid = file.split('.')[0].split('_')[0].upper()
            chain = file.split('.')[0].split('_')[1]
            data = {}
            with open(os.path.join(self.path, 'aa_resolution_dms/dms_{}_{}.txt'.format(pdbid, chain)), 'w') as ofile:
                with open(os.path.join(self.dms_path, file), 'r') as ifile:
                    head = ifile.readline().strip().split('\t') # skip header
                    seq = ifile.readline().strip()
                    ifile.readline() # skip wt entry
                    # read data:
                    for line in ifile:
                        line = line.strip().split('\t')
                        for (score, name) in zip(line[1:], head[1:]):
                            try:
                                score = float(score)
                            except:
                                continue
                            muts = line[0].split(',')
                            for mut in muts:
                                if not '-' in mut:
                                    continue
                                pos, aa = mut.split('-')
                                try:
                                    pos = int(pos)
                                except:
                                    continue
                                if pos in data.keys():
                                    if aa in data[pos].keys():
                                        if name in data[pos][aa].keys():
                                            data[pos][aa][name].append(score)
                                        else:
                                            data[pos][aa][name] = [score]
                                    else:
                                        data[pos][aa] = {name:[score]}
                                else:
                                    data[pos] = {aa: {name:[score]}}

                # calculate per position variance:
                variances = {}
                means = {}
                for pos in data.keys():
                    variances[pos] = {}
                    means[pos] = {}

                    values = {}
                    for aa in data[pos].keys():
                        for name in head[1:]:
                            if name in values:
                                values[name] = values[name] + data[pos][aa][name]
                            else:
                                values[name] = data[pos][aa][name]

                    for name in head[1:]:
                        variances[pos][name] = float(np.var(values[name]))
                        means[pos][name] = float(np.mean(values[name]))

                # write to masked_dump like file
                ofile.write('Pos\tAA')
                for name in head[1:]:
                    ofile.write('\tvar_{}\tmean_{}'.format(name, name))
                ofile.write('\n')
                ofile.write('-1\twt')
                for _ in head[1:]:
                    ofile.write('\t0.0\t0.0')
                ofile.write('\n')

                for i in range(len(seq)):
                    content = [i, seq[i]]
                    for name in head[1:]:
                        try:
                            content = content + [variances[i][name], means[i][name]]
                        except:
                            content = content + ['nan', 'nan']
                    content = [str(c) for c in content]
                    ofile.write('{}\n'.format('\t'.join(content)))
                ofile.flush()

    def fuse_masked_dumps(self, p1, p2, p3, prefix='', check_mw=False):
        """
        Fuses to masked dump files and writes the fusion to a third one.
        :param p1:          path to first file to fuse
        :param p2:          path to second file to fuse
        :param p3:          path to write fusion
        :param prefix:      prefix to add in sedond files header
        :param check_mw:    unused, can check if they have the same mask-width, always 1 here
        :return:            list of pdb id - chain id combinations for which it was successful
        """
        worked = []
        os.makedirs(p3, exist_ok=True)
        p2_files = [f for f in os.listdir(p2) if f.endswith('.txt')]
        p1_files = [f for f in os.listdir(p1) if f.endswith('.txt')]

        for file1 in p1_files:
            f1_split = file1.split('.')[0].split('_')
            pdbid = f1_split[1]
            chain = f1_split[2]
            if len(f1_split) > 3:
                mw = f1_split[3]
            else:
                mw = ''

            warn_pos = None
            fused = False
            if check_mw:
                recognition = '{}_{}_{}'.format(pdbid, chain, mw)
            else:
                recognition = '{}_{}'.format(pdbid, chain)
            opath = os.path.join(p3, 'fused_{}_{}_{}.txt'.format(pdbid, chain, mw))
            for file2 in p2_files:

                if recognition in file2:
                    try:
                        with open(os.path.join(p1, file1), 'r') as ifile1:
                            with open(os.path.join(p2, file2), 'r') as ifile2:
                                with open(opath, 'w') as ofile:
                                    # head:
                                    f1_head = ifile1.readline().strip().split('\t')
                                    f2_head = ifile2.readline().strip().split('\t')
                                    f2_head.remove('Pos')
                                    f2_head.remove('AA')
                                    try:
                                        f2_head.remove('sec')
                                        f2_head.remove('dis')
                                        cut_first = 4
                                    except:
                                        cut_first = 2

                                    f2_head = [prefix + x for x in f2_head]

                                    ofile.write('{}\n'.format('\t'.join(f1_head + f2_head)))

                                    for (line1, line2) in zip(ifile1, ifile2):
                                        if line1.strip() == '' or line2.strip == '':
                                            break
                                        split1 = line1.strip().split('\t')
                                        pos1   = split1[0]
                                        aa1    = split1[1]

                                        split2 = line2.strip().split('\t')
                                        pos2   = split2[0]
                                        aa2    = split2[1]

                                        if aa1 != aa2: # or pos1 != pos2:
                                            warn_pos = pos1
                                            break

                                        ofile.write('{}\n'.format('\t'.join(split1 + split2[cut_first:])))
                    except:
                        self.logger.warning('Fusing files {} and {} failed: {}'.format(file1, file2, sys.exc_info()[0]))
                    fused = True
                    worked.append('{}_{}'.format(pdbid, chain))

                    if warn_pos:
                        self.logger.warning('At position {} in file 1 ({}) a difference with file 2 ({}) was detected. '
                                            'File 1 was copied without changes'
                                            '\n\nline1:\n{}\n\nline2:\n{}'.format(warn_pos, file1, file2, line1, line2))
                        copyfile(os.path.join(p1, file1), os.path.join(p3,
                                                                       'fused_{}_{}_{}.txt'.format(pdbid, chain, mw)))
                    continue

            if not fused:
                copyfile(os.path.join(p1, file1), os.path.join(p3, 'fused_{}_{}_{}.txt'.format(pdbid, chain, mw)))
                pass
        return worked

    def read_masked_dump(self, num_datapoint=0, filename=None):
        """
            Reads the information from one file, returns it in accessible format.
            :param num_datapoint:   index of file in self.masked_dump_files list
            :param filename:    alternative Name of the file
            :return:            the data as a dict: keys are headers of the file, values are lists corresponding to the lines,
                                the pdb identifier, the chain identifier, the mask-width (always 1 in this study)
        """
        self.masked_dump_files = [x for x in os.listdir(self.masked_dump_path) if x.endswith('.txt')]
        count = 0
        if filename:
            file_path = os.path.join(self.masked_dump_path, filename)
            split = filename.split('.')[0].split('_')
            if not os.path.isfile(file_path):
                return False, filename, False, False
        else:
            file_path = os.path.join(self.masked_dump_path, self.masked_dump_files[num_datapoint])
            split = self.masked_dump_files[num_datapoint].split('.')[0].split('_')
        current_id = split[1]
        chain = split[2]
        if len(split) > 3:
            mw = split[3]
        else:
            mw = ''
        with open(file_path, 'r') as ifile:
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
            return False, current_id, chain, mw

        return data, current_id, chain, mw

    def get_max_no(self, path=None):
        """
        Gets the number of files in the masked_dump directory to allow indexing the files
        :param path:    can override the masked_dump directory as target
        :return:        int, the number of files
        """
        if not path:
            path = self.masked_dump_path
        return len([x for x in os.listdir(path) if x.endswith('.txt')])

    def corr(self, x, y):
        """
        Calculate Corr for x vs y.
        :param x: input x
        :param y: input y
        :return: r2, p, n
        """
        x = [float(x_) for x_ in x]
        y = [float(y_) for y_ in y]

        x = np.asarray(x)
        y = np.asarray(y)
        idx = np.isfinite(x) & np.isfinite(y)
        x = x[idx]
        y = y[idx]

        if len(x) == 0:
            return float('nan'), float('nan'), float('nan')

        r2, p = stats.pearsonr(x, y)
        n = len(x)

        return r2, p, n

    def plot_corr(self, x, y, xlabel, ylabel, title='', go=None):
        """
        Get a plot of corr X vs Y.
        :param x: `Array` x-Axis param
        :param y: `Array` y-Axis param
        :param xlabel: 'str' label
        :param ylabel: 'str' label
        :param title: 'str' label
        :param go: 'str' which go
        :return:
        """
        x = [float(x_) for x_ in x]
        y = [float(y_) for y_ in y]

        x = np.asarray(x)
        y = np.asarray(y)
        idx = np.isfinite(x) & np.isfinite(y)
        x = x[idx]
        y = y[idx]

        same = True
        for el in x:
            if el != [x[0]]:
                same = False

        if not same:
            fit = np.polyfit(x, y, 1)
            fit_fn = np.poly1d(fit)

        r2, p = stats.pearsonr(x, y)
        n = len(x)

        plot = not go# or abs(r2) > 0.7

        if plot:
            fig, ax = plt.subplots(figsize=(2, 2))

            plt.scatter(x=x,
                        y=y,
                        alpha=0.5,
                        s=4,
                        c=colors['blue'])

        x = [min(x) - 0.1 * (max(x) - min(x)), max(x) + 0.1 * (max(x) - min(x))]
        y = [min(y) - 0.1 * (max(y) - min(y)), max(y) + 0.1 * (max(y) - min(y))]
        # check if all entries are the same:

        if same:
            fit_line = [np.mean(x)] * 2
        else:
            fit_line = fit_fn(x)
        if plot:
            plt.plot(x, fit_line, c=colors['blue'], label='lin fit')
            plt.xlim(x)
            plt.ylim(y)

            ax.set_title('{} r2={:.2f}, p={:.1E}, n={}'.format(title, r2, p, n))

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            plt.savefig(os.path.join(self.path, 'plots/{}_{}_{}.png'.format(go, ylabel, xlabel)))
            plt.close(fig)

        return p, r2, n

    def align(self, path, chain, seq):
        """
    Basic alignment function to align the sequence in the masked_dump file to the sequence in PDB. Usually
    this is only an issue at the end and at the beginning.

    IMPORTANT: This only works for InDels, not for substitutions as these are uncommon between PDB/SwissProt

    :param path:    Path to PDB file
    :param chain:   Chain identifier
    :param seq:     Sequence from masked dump
    :return:        tuple of
        offset,                 offset of masked dump sequence relative to pdb file
        mapping,                dict that provides the index mapping
        mismatch_percentage,    mismatch percentage
        sequence_from_pdb       the sequence from the pdb file
    """
        seq_pdb = ''
        resi_with_name = {}
        resi_indices = []  # ordered
        with open(path, 'r') as ifile:
            for line in ifile:
                if line[:5] == 'ATOM ':
                    if line[21] == chain:
                        resi = line[23:27].strip()
                        if not resi in resi_indices:
                            resname = line[17:20]
                            resi_with_name[resi] = resname
                            resi_indices.append(resi)
                            seq_pdb += self.three2single[resname]
                        if line[26] != ' ':
                            pass
        seq_len = len(seq)
        resi_len = len(resi_with_name)
        offset_candidates = []

        for offset in range(resi_len + seq_len - 1):
            # seq stays fix: 0 is 0
            offset -= seq_len
            same = True
            mapping = {}  # zero-based seq to resi with insertion codes

            if offset > 0:
                idx_on_seq = 0
                idx_on_resi = abs(offset)  # +int(resi_min)
            else:
                idx_on_seq = abs(offset)
                idx_on_resi = 0  # int(resi_min)

            matches = 0
            mismatches = 0
            while True:
                try:
                    if seq[idx_on_seq] != self.three2single[resi_with_name[resi_indices[idx_on_resi]]]:
                        # tolerating 5 % mismatches.
                        same = ((100 * mismatches / (matches + mismatches) if matches + mismatches > 0 else 100) < 5)
                        mismatches += 1
                    else:
                        mapping[idx_on_seq] = resi_indices[idx_on_resi]
                        matches += 1
                except (KeyError, IndexError) as e:
                    break

                idx_on_resi += 1
                idx_on_seq += 1

            if same:
                mismatch_percentage = 100 * mismatches / (matches + mismatches) if matches + mismatches > 0 else 100
                offset_candidates.append((offset, mapping, mismatch_percentage, seq_pdb))

        longest_mapping = len(offset_candidates[0][1])
        ret = offset_candidates[0]
        for cand in offset_candidates:
            if len(cand[1]) > longest_mapping:
                longest_mapping = len(cand[1])
                ret = cand
        return ret

    def assign_sensitivity(self, structure, chain, sensitivity, aas, pdbpath):
        """
        Matches sensitivity to 3D structure. Takes structure object and assigns each residue a dict with all sensitivity
        information for that residue

        :param structure:   the structure object to which the information is added
        :param chain:       the chain to which the information is added
        :param sensitivity: dict, keys are usually GOs, values are list of sensitivity values
        :param aas:         sequence that matches the sensitivity values, taken from the masked_dump file
        :param pdbpath:     path to the respective PDB file
        :return:            structure object, which has the sensitivitesassigned to its residues as .sensitivity attribute,
        mapping from self.algin()

        """
        try:
            residues = structure[0][chain]
        except KeyError as e:
            self.logger.warning(e)
            return False, False

        offset, mapping, mismatch_percentage, sequence = self.align(pdbpath, chain, aas)

        if len(mapping) < 50:
            self.logger.warning('Had a problem with chain {} of {},'
                                ' most likely because of deletions.'
                                '\n\nPDB\n{}\n\nMasked_Dump:\n{}'.format(chain, pdbpath, sequence, ''.join(aas)))
            return False, False
        self.logger.debug('Tolerated {:.1f} '
                          '% mismatches in chain {} of {}.'.format(mismatch_percentage, chain, pdbpath))

        num_assigned = 0
        for seq_idx in range(len(aas)):
            if seq_idx in mapping:
                try:
                    res_id = mapping[seq_idx]
                    if res_id[-1] in '0123456789':
                        res_id = int(res_id)
                        insertion_code = ' '
                    else:
                        insertion_code = res_id[-1]
                        res_id = int(res_id[:-1])
                    try:
                        residues[(' ', res_id, insertion_code)].sensitivity = {}
                    except KeyError:
                        continue
                except AttributeError:
                    continue
                for go in sensitivity.keys():
                    try:
                        imp = sensitivity[go][seq_idx]
                    except IndexError:
                        imp = float('nan')

                    residues[(' ', res_id, insertion_code)].sensitivity[go] = imp
                num_assigned += 1
        return structure, mapping

    def calc_sphere_variance(self, structure, chain, gos, aas, mapping, ofile):
        """
        Calculate the sphere variance for the specified structure. Also calculated number of neighbours and distance
        to the center of mass for each residue.
        :param structure:   structure object with assigned sensitivity values
        :param chain:       chain in which the sensitivity values are assigned
        :param gos:         GOs associated with that sequence
        :param aas:         sequence as in the masked_dump file
        :param mapping:     mapping from self.align()
        :param ofile:       open file to which the sphere variance is written. has the masked_dump format
        :return:            returns mean sphere variances for each GO
        """
        def dist(a, b):
            return  ((a[0]-b[0])**2
                    +(a[1]-b[1])**2
                    +(a[2]-b[2])**2
                    )**0.5

        center_of_mass = self.calc_center_of_mass(structure, chain)

        residues = structure[0][chain]

        ofile.write('{}\n'.format('\t'.join(['Pos', 'AA', 'n_neighbours', 'd_center'] + gos)))

        variances = {go: 0 for go in gos}
        n_clean = 0
        for seq_idx in range(len(aas)):
            content = [str(seq_idx-1), aas[seq_idx]]
            if seq_idx == 0:
                content += ['nan', 'nan'] + ['0.0' for _ in gos]
                ofile.write('{}\n'.format('\t'.join(content)))
                continue
            if not seq_idx in mapping:
                content += ['nan', 'nan'] + ['nan' for _ in gos]
                ofile.write('{}\n'.format('\t'.join(content)))

            else:
                res_id = mapping[seq_idx]
                if res_id[-1] in '0123456789':
                    res_id = int(res_id)
                    insertion_code = ' '
                else:
                    insertion_code = res_id[-1]
                    res_id = int(res_id[:-1])
                try:
                    res = residues[(' ', res_id, insertion_code)]
                    ca = res['CA']

                except:
                    content += ['nan', 'nan'] + ['nan' for _ in gos]
                    ofile.write('{}\n'.format('\t'.join(content)))
                    continue

                glob_het_atm, glob_resseq, glob_icode = res.get_id()

                center = ca.get_coord()
                search = pdb.NeighborSearch(atom_list=list(structure.get_atoms()))
                neighbors = search.search(center=center, radius=self.radius, level="R")

                clean_neighbors = []
                for n in neighbors:
                    het_atm, resseq, icode = n.get_id()
                    if het_atm.strip() == '' and resseq >= 0:
                        if abs(resseq - glob_resseq) > self.exclude:
                            if hasattr(n, 'sensitivity'):
                                clean_neighbors.append(n)
                if len(clean_neighbors) == 0:
                    content += ['0', str(dist(center_of_mass, ca.get_coord()))] + ['nan' for _ in gos]
                    ofile.write('{}\n'.format('\t'.join(content)))
                    continue

                imps = {x:[] for x in gos}

                for n in clean_neighbors:
                    for go in gos:
                        try:
                            imps[go].append(n.sensitivity[go])
                        except AttributeError:
                            pass
                content.append(str(len(clean_neighbors)))
                content.append(str(dist(center_of_mass, ca.get_coord()))) # distance to center of mass

                for go in gos:
                    current_var = np.var(imps[go])
                    content.append(str(current_var))
                    variances[go] += current_var
                    n_clean += 1
                ofile.write('{}\n'.format('\t'.join(content)))

        if n_clean > 1:
            for go in gos:
                variances[go] /= n_clean
        return variances

    def calc_center_of_mass(self, structure, chain):
        """
        Calculate the center of mass of the given chain in the given structure
        :param structure:   the structure object in wich the center of mass is calculated
        :param chain:       the chain of which the center of mass is calculated
        :return:            the coordinates of the center of mass
        """
        # http://en.wikipedia.org/wiki/List_of_elements
        masses = {'H':1.008, 'C':12.011, 'N':14.007, 'O':15.999, 'S':32.06}
        residues = structure[0][chain]
        weight_coordinates = []
        for resi in residues:
            het_atm, resseq, icode = resi.get_id()
            if het_atm.strip() != '':
                continue
            for atom in resi:
                weight_coordinates.append([atom.get_coord(), masses.get(atom.element, 0.0)])

        full_weight = sum([weight_coordinates[n][1] for n in range(len(weight_coordinates))])

        center = [sum([weight_coordinates[n][0][dim] * weight_coordinates[n][1]/full_weight
                       for n in range(len(weight_coordinates))]) for dim in range(3)]
        return center


    def write_3d(self, files=None):
        """
        Assigns sensitivity values to all proteins analyzed. Calculates the sphere variance and produces the corresponding
        masked_dump files
        :param files: a list of files can be specified. If it is not specified, the current masked_dump path is used.
        :return: None
        """
        if not files:
            pdb_order = [x for x in os.listdir(self.masked_dump_path) if x.endswith('.txt')]
        else:
            pdb_order = files
        count = 0
        total = len(pdb_order)
        for filename in pdb_order: # this preserves order
            count += 1
            data, pdbid, chain, mw = self.read_masked_dump(filename=filename)
            if not data:
                self.logger.debug('({:.1f} %) File not found or broken: {}'.format(100*count/total, filename))
                continue

            outpath = os.path.join(self.path_3d, '3d_{}_{}_{}.txt'.format(pdbid, chain, mw))
            if os.path.isfile(outpath):
                self.logger.debug('({:.1f} %) File already existed: {}'.format(100*count/total, outpath))
                continue
            with open(outpath, 'w') as _: # block this file
                pass

            path = os.path.join(self.pdbpath, '{}.pdb'.format(pdbid))
            # self.logger.debug('Processing chain {} of {} with mask width {} now.'.format(chain, path, mw))
            structure = self.parser.get_structure(id='{}_{}'.format(pdbid, chain), file=path)
            if not structure:
                self.logger.warning('({:.1f} %) Could not get structure for pdb id {}'.format(100*count/total, pdbid))
                try:
                    os.remove(outpath)
                except:
                    self.logger.warning('Could not remove 3d file for pdb id {}_{}'.format(pdbid, chain))
                continue

            gos = []
            # get all relevant gos
            for key in data.keys():
                if key.startswith('GO:') and key[-2] != '_':
                    gos.append(key)

            sensitivity = {}
            for go in gos:
                sensitivity[go] = data[go][1:]

            # real values
            tmp, mapping = self.assign_sensitivity(structure, chain, sensitivity, data['AA'], path)
            if not tmp:
                seq = ''
                for (aa, dis) in zip(data['AA'], data['dis']):
                    if dis != 'X':
                        seq += aa
                structure, mapping = self.assign_sensitivity(structure, chain, sensitivity, data['AA'], path)
            else:
                structure = tmp
            if not structure:
                self.logger.warning('({:.1f} %) Structure {} {} has a problem.'.format(100*count/total, pdbid, chain))
                try:
                    os.remove(outpath)
                except:
                    pass
                continue
            with open(outpath, 'w') as ofile:
                self.logger.debug('({:.1f} %) File looks ok: {}'.format(100*count/total, filename))
                self.calc_sphere_variance(structure, chain, gos, data['AA'], mapping, ofile)

    def plot_dist(self, data, x_title, y_title, title, name=None, log=False):
        """
        Plots a distribution of given values
        :param data:        list, the data of which the distribution is plotted
        :param x_title:     X axis title for plot
        :param y_title:     Y axis title for plot
        :param title:       Title for plot
        :param name:        filename for plot
        :param log:         Enables logarithmic y-scale
        :return:            None
        """
        if not name:
            name = title
        fig, ax = plt.subplots()
        data = [d for d in data if not np.isnan(d)]
        data = sorted(data, key=lambda x: -x)

        ax.bar(x=list(range(len(data))),
               height=data,
               width=1,
               color=colors['blue'],
               alpha=0.7
               )

        if not log:
            ax.set_ylim([0, 1])
            ax.set_yticks(     [0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([0, len(data)])
        ax.set_xticklabels([0, len(data)])
        ax.set_ylabel(y_title)
        ax.set_xlabel(x_title)
        ax.set_title(title)
        if log:
            ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(name)))
        plt.close()

    def analyse_sequences(self):
        """
        Plots the distribution of all correlations between information content and sensitivity. Uses the sequence table
        files.
        :return: None
        """

        files = [f for f in os.listdir(self.seq_resolution) if f.endswith('.txt')]
        data = {}
        for file in files:
            with open(os.path.join(self.seq_resolution, file), 'r') as ifile:
                head = [h.strip() for h in ifile.readline().strip().split(';')]
                for line in ifile:
                    for (entry, category) in zip(line.strip().split(';'), head):
                        try:
                            entry = float(entry)
                        except:
                            pass
                        if not category in data:
                            data[category] = [entry]
                        else:
                            data[category].append(entry)

        self.plot_dist(data['cor_ic'], 'Sequences', 'r2: {}'.format('cor_ic'), 'Distribution of Correlations'.format('cor_ic'), 'cor_ic')


    def analyse_go_correlations(self, go_correlations):
        """
        Analyzes correlations between GO terms on the same sequence. Finds out which GO terms are connected by an
        'is a' (i.e. parent-child) relationship, reports their correlation. Makes boxplot of the r2 values per GO level of the parent.
        :return: nested list: each entry is for a pair of GO terms connected by an 'is a' relationship, each entry features
            both GO terms,
            the level of the parent term,
             the number of points for which the correlation was calculated,
             the mean r2 value
             the mean p value
             all r2 values as a list
             all p values as a list
        """

        data = []
        for go1 in go_correlations:
            for go2 in go_correlations:
                go1_obj = self.GODag.query_term(go1)
                if go1_obj.has_child(go2):
                    data.append([  go1,
                                   go2,
                                   go1_obj.level,
                                   len(go_correlations[go1][go2]['r2']),
                                   float(np.nanmean(
                                       go_correlations[go1][go2]['r2'])) if len(
                                       go_correlations[go1][go2]['r2']) != 0 else float('nan'),
                                   float(np.nanmean(
                                       go_correlations[go1][go2]['p'])) if len(
                                       go_correlations[go1][go2]['p' ]) != 0 else float('nan'),
                                   go_correlations[go1][go2]['r2'],
                                   go_correlations[go1][go2]['p']
                                   ])

        with open(os.path.join(self.path, 'go_parent-child_correlations.txt'), 'w') as ofile:
            ofile.write('{}\n'.format('\t'.join(['Parent', 'Child', 'parent_level','n', 'mean_r2',
                                                 'mean_p', 'comma_joined_r2_values', 'comma_joined_p_values'])))
            for line in data:
                ofile.write('{}\n'.format(
                    '\t'.join([str(l) for l in line[:6]] + [';'.join(
                        [str(l) for l in line[6]]), ';'.join([str(l) for l in line[7]])])))

        fig, ax = plt.subplots()

        levels = []
        [levels.append(x[2]) for x in data if not x[2] in levels]
        levels = sorted(levels)
        bp_data = [[x[4] for x in data if x[2] == l and not np.isnan(x[4])]for l in levels]

        ax.boxplot(bp_data,
                   labels=levels,
                   flierprops={'markersize': 2},
                   medianprops={'color': colors['blue']})
        ax.set_ylabel('r2')
        ax.set_xlabel('GO level')
        ax.set_title('GO pair correlations')
        for l in range(len(bp_data)):
            plt.text(x=l+1,
                     y=-0.9 if l%2==0 else -0.8,
                     s=len(bp_data[l]),
                     horizontalalignment='center',
                     verticalalignment='center'
            )
        plt.xlim([-0.1, len(bp_data) + 1])
        ax.set_yticks([-0.84, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(['n{', -0.5, 0, 0.5, 1])
        plt.ylim([-1, 1.1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'go_pair_correlations-level.png'))
        plt.close()

        return data

    def investigate_go_correlations(self, gos):
        """Reads all per aa-resolution files and calculated all correlations between the impact of the GO terms.
        Makes a 1000x1000 Matrix containing all correlations between two GOs as lists of r2 and p values"""
        data = {}  # ['GO:1']['GO:2'] = {'p': [1, 2, 3, ...], 'r2': [4, 5, 6, ...]}
        for go1 in gos:
            data[go1] = {}
            for go2 in gos:
                data[go1][go2] = {'p': [], 'r2': []}

        path = self.masked_dump_path

        self.max_num_datapoints = self.get_max_no(path)
        self.logger.debug(
            'Found {} datapoints for investigating Correlations betwenn GOs'.format(self.max_num_datapoints))

        for datapoint in range(self.max_num_datapoints):
            per_aa_data, current_id, chain, mw = self.read_masked_dump(datapoint)
            if not per_aa_data:
                self.logger.warning('File for {} {} is invalid'.format(current_id, chain))
                continue
            gos_entry = []
            # get all relevant gos
            for key in per_aa_data.keys():
                if key.startswith('GO:') and key[-2] != '_':
                    gos_entry.append(key)

            for go1 in gos_entry:
                for go2 in gos_entry:
                    if go1 == go2:
                        r2 = 1.0
                        p = 0.0
                    else:
                        r2, p, n = self.corr(per_aa_data[go1][1:], per_aa_data[go2][1:])

                    data[go1][go2]['p'].append(p)
                    data[go1][go2]['r2'].append(r2)
                    data[go2][go1]['p'].append(p)
                    data[go2][go1]['r2'].append(r2)
        return data

    def write_table_seq(self, path=None):
        """
        Writes a table for each GO term. Each line is one sequence, summaries all information available at that level.
        Most importantly this includes the number of OG children of the GO term, the number of GO terms for that sequence,
        the sequence length, and the correlation with the information conteent, if available for the sequence.
        :param path: can overwrite current masked_dump path
        :return: None
        """

        if not path:
            path = self.masked_dump_path
        self.logger.debug('Writing sequence table from {}'.format(path))

        pdb2pfam = {}
        pdb_order = []
        with open(self.pfam_path, 'r') as ifile:
            for line in ifile:
                line = line.strip().split(';')
                pdb2pfam['{}_{}'.format(line[1], line[2])] = line[0]
                pdb_order.append('{}_{}'.format(line[1], line[2]))

        self.max_num_datapoints = self.get_max_no(path)
        self.logger.debug('Found {} datapoints for writing the sequence table.'.format(self.max_num_datapoints))

        # data, current_id, chain, mw = self.read_masked_dump(0)
        open_files = {}  # keeps all files
        cats = ['ID'] \
               + ['stt_len', 'stt_nGO_children', 'stt_n_gos'] \
               + ['cor_ic']

        # parser for pdb files: 3d
        os.makedirs(os.path.join(self.path, 'data3d'), exist_ok=True)

        for datapoint in range(self.max_num_datapoints):
            data, current_id, chain, mw = self.read_masked_dump(datapoint)

            if not data:
                self.logger.warning('File for {} {} is invalid'.format(current_id, chain))
                continue
            pfam = pdb2pfam.get('{}_{}'.format(current_id, chain), 'PFXXXXX')

            gos = []
            # get all relevant gos
            for key in data.keys():
                if key.startswith('GO:') and key[-2] != '_':
                    gos.append(key)
                    self.gos.add(key)

            seq_len = (len(data['Pos']) - 1)

            # get the data
            for go in gos:

                content = []
                for cat in cats:
                    if cat == 'ID':
                        content.append('{}_{}_{}'.format(current_id, chain, mw))  # unique
                    elif cat == 'stt_len':
                        content.append(seq_len)
                    elif cat == 'stt_nGO_children':
                        try:
                            content.append(len(self.GODag.query_term(go).get_all_children()))
                        except IndexError:
                            content.append('nan')
                    elif cat == 'stt_n_gos':
                        content.append(len(gos))

                    elif cat == 'cor_ic':
                        content.append(self.corr(data[go][1:],
                                                 data['ic'][1:])[0]
                                       if 'ic' in data else 'nan')
                    else:
                        self.logger.warning('key {} from head has no data.'.format(cat))
                        content.append('nan')
                content = [str(x) for x in content]

                if not go in open_files:
                    open_files[go] = open(os.path.join(self.seq_resolution, 'seq_report_{}.txt'.format(go)), 'w')
                    open_files[go].write('{}\n'.format('; '.join(cats)))

                open_files[go].write('{}\n'.format('; '.join(content)))
                open_files[go].flush()

        # shutdown
        [open_files[go].close() for go in open_files]

    def run(self):
        """
        Run analysis.
        """
        self.masked_dump_path = os.path.join(self.path, 'aa_resolution/')
        self.path_3d = os.path.join(self.path, 'aa_resolution_3d')
        self.seq_resolution = os.path.join(self.path, 'seq_resolution')

        os.makedirs(self.path_3d, exist_ok=True)
        os.makedirs(self.seq_resolution, exist_ok=True)
        # extend masked dump by dms data:

        self.parser = pdb.PDBParser()

        self.logger.debug('Writing 3d')
        self.write_3d()

        self.logger.debug('Fusing variance')
        if os.path.exists(os.path.join(self.path, 'aa_resolution_variance')):
            success_variance = self.fuse_masked_dumps(self.masked_dump_path,
                                                 os.path.join(self.path, 'aa_resolution_variance'),
                                                 os.path.join(self.path, 'aa_resolution_with_variance'),
                                                 check_mw=False)
            self.masked_dump_path = os.path.join(self.path, 'aa_resolution_with_variance')
            self.logger.debug('Fused variance to test.')
        else:
            self.logger.debug('Did not find variance dumps')

        self.logger.debug('Fusing files')
        success_3d  = self.fuse_masked_dumps(os.path.join(self.masked_dump_path),
                                             self.path_3d,
                                             os.path.join(self.path, 'aa_resolution_test_with_3d'),
                                             prefix='svar_',
                                             check_mw=True)
        self.masked_dump_path = os.path.join(self.path, 'aa_resolution_test_with_3d')
        self.logger.debug('Fused 3d to test.')

        if not os.path.exists(os.path.join(self.path, 'aa_resolution_dms')):
            self.logger.debug('Counting dms.')
            self.count_dms()

        if os.path.exists(os.path.join(self.path, 'aa_resolution_dms')):
            success_dms = self.fuse_masked_dumps(self.masked_dump_path,
                                                 os.path.join(self.path, 'aa_resolution_dms'),
                                                 os.path.join(self.path, 'aa_resolution_test_with_dms'),
                                                 check_mw=False)
            self.masked_dump_path = os.path.join(self.path, 'aa_resolution_test_with_dms')
            self.logger.debug('Fused dms to test.')
        else:
            self.logger.debug('Did not find DMS dumps')

        if self.ic_path:
            self.logger.debug('Fusing IC to masked_dump')
            success_ic  = self.fuse_masked_dumps(self.masked_dump_path,
                                                 self.ic_path,
                                                 os.path.join(self.path, 'aa_resolution_test_with_ic'),
                                                 check_mw=False)
            self.masked_dump_path = os.path.join(self.path, 'aa_resolution_test_with_ic')
            self.logger.debug('Fused ic to test.')

        self.masked_dump_path = os.path.join(self.path, 'aa_resolution_test_with_ic')

        self.write_table_seq()

        self.logger.debug('Analysis on sequence level')
        self.analyse_sequences()

        go_correlations = self.investigate_go_correlations(self.gos)

        self.analyse_go_correlations(go_correlations)

    def __init__(self, path, name):
        self.path = path

        self.plot_path = os.path.join(path, 'plots')
        os.makedirs(self.plot_path, exist_ok=True)
        self.logger = self.set_up_logger(path, name)
        self.masked_dump_files = None

        self.gos = set()

        self.aas3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        self.aas1 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.three2single = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        self.radius = 8.0  # in angstrom
        self.exclude = 4  # number of residues following and preceeding the center residue to exclude from stdev calculation
        self.bs = 100  # number of permutations of sensitivity scores to test against


        self.dms_path = 'path to dms datasets'
        self.ic_path = 'path to dumps with information content'
        self.pfam_path = 'path to pfam2go.json file, mapping pfam identifiers to lists of go terms'
        self.pdbpath = 'path to pdb files'
        go_dag_file = 'path to go dag'

        self.GODag = GODag(go_dag_file, optional_attrs=['relationship'])

        self.run()
        with open(os.path.join(self.path.replace(self.path.split('/')[-1], ''), 'DONE_EVALUATE_MD.txt'), 'w') as _:
            pass
        self.logger.debug('DONE.')


if __name__ == '__main__':
    path = sys.argv[1]
    try:
        name = sys.argv[5]
        if name == '_':
            name = 'evaluate'
    except:
        name = 'evaluate'

    _ = evaluate(path, name)
