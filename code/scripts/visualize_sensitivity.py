from pymol import cmd, stored
import pymol
import numpy as np
import os
import urllib
from random import shuffle

try:
    from goatools.obo_parser import GODag
except:
    GODag = None

"""
This was written for pyMol 1.7.4.5

Enter 'run path/visualize_sensitivity.py' in PyMol where 'path' is replaced by the path of this file. An example command 
is printed in the pyMol console.

Have a look at the color_sensitivity function docstring, which explains the core of this module.

IMPORTANT
Specify the paths below.
"""

label_path          = 'path/to/label'
save_path            = 'path/to/saves'
godag_path          = 'path/to/godag'
prot_path           = 'path/to/pdb-files'

exclude_file_path   = os.path.join(save_path, 'exclude.txt')


def get_pdb_file(path, pdbid):
    """
    Downloads PDB file with specified ID if it is not already present in the specified path. Tries 10 times and gives
    up then.
    :param path:    Path in which the PDB file is first searched, and stored if it is downloaded
    :param pdbid:   PDB id of the file in question
    :return:        True, if successful, false if the file was not found and could not be downloaded
    """
    if not os.path.exists(path):
        print('Downloading PDB file {} to {}'.format(pdbid, path))
        fails = 0
        while True:
            try:
                urllib.urlretrieve('https://files.rcsb.org/download/{}.pdb'.format(pdbid), path)
                break
            except IndexError:
                if fails == 10:
                    print('FAILED')
                    return False
                fails += 1
    return True


def get_name_from_pdb(fpath):
    """
    Extracts the title from a PDB file
    :param fpath:   PDB ID in question
    :return:        The title
    """
    name = ''
    with open(fpath, 'r') as ifile:
        for line in ifile:
            if line.startswith('TITLE'):
                name += line.replace('TITLE', '').strip()
            if line.startswith('COMPND'):
                break
    return name


def get_max_no(file_path):
    """
    Returns the number of files in the label directory
    :param file_path:   Path in which to look, the label directory
    :return:            The number of files that can be used in this directory
    """
    return len([x for x in os.listdir(file_path) if x.endswith('.txt')])


def read_masked_dump(file_path, filename=None):
    """
    Reads the information from one file, returns it in accessible format.
    :param file_path:   Path to the file
    :param filename:    Name of the file
    :return:            the data as a dict: keys are headers of the file, values are lists corresponding to the lines,
                        the pdb identifier, the chain identifier, the mask-width (always 1 in this study)
    """

    split = filename.split('.')[0].split('_')
    current_id = split[1]
    chain = split[2]
    if len(split) > 3:
        mw = split[3]
    else:
        mw = ''

    file = os.path.join(file_path, filename)
    with open(file, 'r') as ifile:
        keys = ifile.readline().strip().split('\t')

        data = {}
        for key in keys:
            data[key.strip()] = []
        for line in ifile:
            if line.startswith('>'):
                break
            for entry, key in zip(line.strip().split('\t'), keys):
                try:
                    data[key.strip()].append(float(entry))
                except:
                    data[key.strip()].append(entry)

    return data, current_id, chain, mw


def get_resi_bounds(path, chain):
    """
    Returns minimum and maximum residue identifier from the pdb file, including only the chain in question
    :param path:    PDB id of the file in question
    :param chain:   Chain id in question
    :return:        Minimum and maximum residue identifier
    """
    min_resi = None
    max_resi = None
    with open(path, 'r') as ifile:
        for line in ifile:
            if line[:5] == 'ATOM ':
                if line[21] == chain:
                    resi = int(line[22:26].strip())
                    if not min_resi and min_resi != 0:
                        min_resi = resi
                        max_resi = resi
                    else:
                        if resi < min_resi:
                            min_resi = resi
                        if resi > max_resi:
                            max_resi = resi
    return min_resi, max_resi


def align(path, chain, seq):
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

    sequence_from_pdb = ''
    three2single = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                    'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    resi_with_name = {}
    resi_indices = [] # ordered
    with open(path, 'r') as ifile:
        for line in ifile:
            if line[:5] == 'ATOM ':
                if line[21] == chain:
                    resi = line[22:27].strip()
                    if not resi in resi_indices:
                        resname = line[17:20]
                        resi_with_name[resi] = resname
                        resi_indices.append(resi)
                        sequence_from_pdb += three2single[resname]
                    if line[26] != ' ' :
                        pass

    seq_len = len(seq)
    resi_len = len(resi_with_name)
    offset_candidates = []
    for offset in range(resi_len + seq_len -1):
        offset -= seq_len
        same = True
        mapping = {} # zero-based seq to resi with insertion codes

        if offset > 0:
            idx_on_seq = 0
            idx_on_resi =  abs(offset)
        else:
            idx_on_seq = abs(offset)
            idx_on_resi = 0
        matches = 0
        mismatches = 0
        while True:
            try:
                if seq[idx_on_seq] != three2single[resi_with_name[resi_indices[idx_on_resi]]]:
                    same = ((100 * mismatches / (matches + mismatches) if matches + mismatches > 0 else 100) < 5)
                    # tolerating 5 % mismatches
                    mapping[idx_on_seq] = resi_indices[idx_on_resi]

                    mismatches += 1
                else:
                    mapping[idx_on_seq] = resi_indices[idx_on_resi]
                    matches += 1
            except (KeyError, IndexError) as e:
                break

            idx_on_resi += 1
            idx_on_seq  += 1

        if same:
            mismatch_percentage = 100 * mismatches / (matches + mismatches) if matches + mismatches > 0 else 100
            offset_candidates.append((offset, mapping, mismatch_percentage, sequence_from_pdb))

    ret = offset_candidates[0]
    old_length = len(ret[1])
    for cand in offset_candidates:
        new_length = len(cand[1])
        if new_length > old_length:
            old_length = new_length
            ret = cand

    return ret


def reset_stored():
    """
    Used to reset the values stored in PyMol.
    :return: None
    """
    stored.current_no_go = 0
    stored.last_pdbid = ''

    stored.order = os.listdir(label_path)

    excl = [l.strip() for l in open(exclude_file_path, 'r').readlines()]

    for e in excl:
        if e in stored.order:
            stored.order.remove(e)

    shuffle(stored.order)

    stored.exclude_file = open(exclude_file_path, 'a')

    stored.current_file = stored.order.pop()

    stored.exclude_file.write(stored.current_file + '\n')


def picture(name):
    """
    Takes a picture, stores it under the spcified name in the savepath
    :param name: name for the image file
    """
    cmd.png(os.path.join(save_path, name),
            width=2304,
            height=1440,
            dpi=300,
            ray=0,
            quiet=0)

def color_sensitivity(save_png=False, next_dp=False, file=None, category=None,
                 only=None, show_hetatm=True, show_chains=False, lig=None):
    """
    Use to automatically color a structure by sensitivity. Is called from the PyMol console.

    :param save_png:        if True saves a PNG of the current view at the next call. IMPORTANT: the current view will
                            only be saved at the next call.
    :param next_dp:         jumps to next file at the next call. Conflicts with the 'file' parameter, always shows the
                            first category then.
    :param file:            filename of the label file to use. None selects one from the specified label path
    :param category:        column to color the structure with. None selects one from the current file
    :param only:            cycles trough columns that contain information to what is specified here, options are:
        a GO term   -> shows the sensitivity for that, the sphere variance, whatever is available
        bind        -> shows sensitivity for GO terms which contain 'bind' in their name
        sen         -> only sensitivity columns
        svar        -> only shows sphere variance columns
    :param show_hetatm:     Wheter to show heteroatoms (like ligands)
    :param show_chains:     Can be False, shows no chains except for the one with sensitvity data then.
                            Can be True, shows all chains, the ones without sensitivity data in orange.
                            Can be a nested list like '[['A', 'orange'], ['C', 'yellow'], ['D', 'yellow']]',
                            each element in the list must be a list with two elements: A single letter specifying the
                            chain and a string that is a valid color for pyMOL.
    :param lig:             Ligand identifier. If specified the distance of all residues to that ligand is calculated
                            and written to the savepath as a new column in the current label file. The ligand identifier
                            can be determined as follows:
                                1. Click on the ligand
                                2. The Console will show something like 'You clicked   /5VW1//C/PG4`102/O4'
                                3. Copy the part from the first to the last '/', like '/5VW1//C/PG4`102/'
                                4. Paste

    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    try:
        stored.last_pdbid
    except AttributeError:
        stored.last_pdbid = 'None'

    try:
        stored.dest
    except AttributeError:
        stored.dest = 'None'

    try:
        stored.godag
    except AttributeError:
        if GODag:
            stored.godag = GODag(godag_path, optional_attrs=['relationship'])
        else:
            stored.godag = None


    if stored.dest != 'None' and save_png:
        cmd.png(stored.dest,
                width=2304,
                height=1440,
                dpi=300,
                ray=0,
                quiet=0)


    if not file:
        data, current_id, chain, mw = read_masked_dump(label_path, stored.current_file)
    else:
        data, current_id, chain, mw = read_masked_dump(label_path, file)
        stored.current_file = file

    print(', '.join(data.keys()))
    head = list(data.keys())

    if not only:
        head = [e for e in head if e not in {'Pos', 'AA', 'sec', 'dis'}]
    else:
        if len(only) == 1: # single aa
            head = [h for h in head if h.endswith('_{}'.format(only))]
        elif only.startswith('GO:'): # show everything about that GO
            head = [h for h in head if only in h]
        elif only=='sen': # only show sensitivity
            head = [h for h in head if h.startswith('GO:') and len(h) == 10]
        elif only=='svar': # only show svar
            head = [h for h in head if h.startswith('svar')]
        elif only=='bind':
            if not stored.godag:
                print('The bind option is not available if no GODag is provided.')
            tmp = []
            for h in head:
                if h.startswith('GO:'):
                    try:
                        name = stored.godag.query_term(h).name
                    except:
                        name = '?'
                    if 'inding' in name:
                        tmp.append(h)
                        print('{}->{}'.format(h, name))
            head = tmp

        else:
            print('{} unknown.'.format(only))

    max_no_go = len(head)

    try:
        stored.current_no_go
    except AttributeError:
        stored.current_no_go = 0

    if next_dp:
        stored.current_file = stored.order.pop()
        stored.exclude_file.write(stored.current_file + '\n')

    no_go = stored.current_no_go

    if no_go >= max_no_go:
        stored.current_no_go = 0
        no_go = stored.current_no_go
        stored.current_file = stored.order.pop()
        stored.exclude_file.write(stored.current_file + '\n')

        print('Moving to the next datapoint.')

    if not category:
        category = head[no_go]
    # open the file of new values (just 1 column of numbers, one for each alpha carbon)
    current_id = current_id.upper()
    pdb_file = os.path.join(prot_path, '{}.pdb'.format(current_id))
    if not get_pdb_file(pdb_file, current_id):
        print('could not get pdb file with id {}.'.format(current_id))

    # load the protein
    if stored.last_pdbid != current_id:
        print('Loading new pdb file')
        cmd.reinitialize()
        cmd.load(pdb_file)

    stored.last_pdbid = current_id

    name = get_name_from_pdb(pdb_file)

    # find out the offset
    min_resi, max_resi = get_resi_bounds(pdb_file, chain)
    offset, mapping, mmpercentage, sequence = align(pdb_file, chain, ''.join(data['AA'][1:]))
    if len(mapping) < 50:
        seq = ''
        for (aa, dis) in zip(data['AA'][1:], data['dis'][1:]):
            if dis != 'X':
                seq += aa
        offset, mapping, mmpercentage, sequence = align(pdb_file, chain, seq)

    if len(mapping) < 50:
        stored.current_no_go = 0
        stored.current_file = stored.order.pop()
        stored.exclude_file.write(stored.current_file + '\n')

        print('Had a problem with showing {}, chain {} for {} mask width {}\n{}'
              '\n\nPDB:\n{}\n\nMasked_dump:\n{}\nMasked_dump_X:\n{}'.format(
            category, chain, current_id, mw, name, sequence, ''.join(data['AA'][1:]), seq))
        return
    print('Tolerated {} % mismatches.'.format(mmpercentage))

    cutoff = max_resi - offset

    # process the data: set min to -1, max to 1, the middle to zero

    tmp = data[category][1:cutoff]
    #tmp = [t if not np.isnan(t) else 0.0 for t in tmp]
    fixed_data = []
    tmp2 = list(tmp)
    tmp2 = [t if not np.isnan(t) else 0.0 for t in tmp2]
    while 0.0 in tmp2:
        tmp2.remove(0.0)
    min_d = min(tmp2)
    max_d = max(tmp2)
    #max_d -= min_d

    if category == 'ic':
        max_d -= min_d
    for val in tmp:
        #val -= min_d
        if category == 'ic':
            val -= min_d
        if val < 0:
            val = -val/min_d if min_d != 0 else val # keep this negative
        else:
            val = val/max_d if max_d != 0 else val
        fixed_data.append(val)

    # clear out the old B Factors
    cmd.select("sele", "all")
    cmd.alter("sele", "b=0.0")

    # put in the new B factors
    max_b = 0

    if lig:
        assert(not next_dp)
        ipath = label_path
        opath = save_path

        ofile_path = os.path.join(opath, stored.current_file)
        print(ofile_path)
        ifile =  open(os.path.join(ipath, stored.current_file), 'r')
        ofile =  open(ofile_path, 'w')

        ofile.write(ifile.readline().strip() + '\td-lig\n')
        ofile.write(ifile.readline().strip() + '\t0.0\n')

    for seq_idx in range(len(fixed_data)):
        if seq_idx in mapping:
            cmd.select("sele", "chain {} & resi \{}".format(chain, mapping[seq_idx]))
            tmp = fixed_data[seq_idx]
            if not np.isnan(tmp):
                cmd.alter("sele", "b={}".format(tmp))
            else:
                cmd.alter("sele", "b={}".format(0.0))

            if lig:
                mean_dist = cmd.distance('({}/*)'.format(lig), 'sele')
                print('{}{}: {}'.format(seq_idx, data['AA'][1:][seq_idx], mean_dist))
                tmp = mean_dist
                cmd.alter("sele", "b={}".format(tmp))
                line = ifile.readline().strip()
                split = line.split('\t')
                no = split[0]
                aa = split[1]
                if not line.startswith('{}\t{}\t'.format(no, aa)):
                    print('Line does not match aa and res no')
                else:
                    ofile.write(line + '\t' + str(tmp) + '\n')

                max_b = tmp if tmp > max_b else max_b
            else:
                max_b = 1.0

    print('Max b: {}'.format(max_b))

    cmd.select("all")
    cmd.hide("everything")
    cmd.select("sele", "chain {}".format(chain))
    cmd.show("cartoon", "sele")
    cmd.color("grey", "sele")

    for seq_idx in range(len(fixed_data)):
        if seq_idx in mapping:
            if not np.isnan(fixed_data[seq_idx]):
                cmd.select("sele", "chain {} & resi \{}".format(chain, mapping[seq_idx]))
                cmd.spectrum("b", "red_white_blue", "sele", "-1.0", "{}".format(max_b))

    cmd.select("sele", "resn HOH")
    cmd.remove("sele")
    if show_hetatm:
        cmd.select("sele", "hetatm")
        cmd.color("yellow", "sele")
        cmd.show("sticks", "sele")
    if show_chains:
        cmd.select("sele", "not chain {}".format(chain))
        cmd.color("orange", "sele")
        cmd.show("cartoon", "sele")
        try:
            for e in show_chains:
                cmd.select("sele", "chain {}".format(e[0]))
                cmd.color(e[1], "sele")
                cmd.show("cartoon", "sele")
        except:
            pass
    cmd.set("overlay")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 'off')
    print('Showing {}, chain {} for {} mask width {}\n{}\nmin {} and max {}.'
          ''.format(category, chain, current_id, mw, name, min_d, max_d))
    if 'GO:' in category:
        if stored.godag:
            go_obj = stored.godag.query_term(category[category.find('GO:'): category.find('GO:') + 10])
            go_name = go_obj.name
            go_level = go_obj.level
        else:
            go_name = '_'
            go_level = ''
    else:
        go_name = '_'
        go_level = ''
    print('{} {} (level {})'.format(category, go_name, go_level))

    cmd.set("seq_view")
    stored.dest = os.path.join(save_path, '{}_{}_{}_{}_{}'
                                         ''.format(current_id, chain, mw, category.replace(':', '-'), go_name))

    cmd.deselect()

    stored.info = {'pdbid': current_id,
                   'chain': chain}
    # update current_no
    stored.current_no_go += 1

    if lig:
        ifile.close()
        ofile.close()

########################################################################################################################

cmd.extend("color_sensitivity", color_sensitivity)
cmd.extend("reset_stored", reset_stored)
cmd.extend("picture", picture)


print('Working with the following directories: \n'
      'labels: {}\n saves: {},\n godag file: {},\n structures: {}\n'.format(label_path, save_path, godag_path, prot_path))

print("\ncolor_sensitivity(save_png=False, next_dp=False, file=None, category=None, only=None,"
      " show_hetatm=False, show_chains=None)\n")