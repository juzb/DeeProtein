import os
import urllib
from goatools.obo_parser import GODag
import random
import sys
import urllib.request


def count_children(GODag, go):
    return len(GODag.query_term(go).get_all_children())


def extend_gos_by_parents(GODag, gos, logger=None):
    gos_out = set()
    for go in gos:
        gos_out.add(go)
        try:
            gos_out.update(GODag.query_term(go).get_all_parents())
        except:
            if logger:
                logger.debug('Could not get parents for term {}.'.format(go))
    return gos_out


def filter_seq(seq):
    for c in seq:
        if not c in 'ACDEFGHIKLMNPQRSTVWY':
            return False
    return True


def filter_gos(gos):
    return True


def calc_secondary(secondary):
    ret = {x: 0 for x in 'HBEGITS.'}  # see http://www.rcsb.org/pages/help/ssHelp
    for c in secondary:
        ret[c] += 1
    return ret


def calc_disorder(disorder):
    ret = {x: 0 for x in 'X-'}
    for c in disorder:
        ret[c] += 1
    return ret


def calc_sequence(sequence):
    ret = {x: 0 for x in 'ACDEFGHIKLMNPQRSTVWY'}
    for c in sequence:
        ret[c] += 1
    return ret


def get_pdb_file(id):
    path = 'pdb_files/{}.pdb'.format(id)
    if not os.path.exists(path):
        fails = 0
        while True:
            try:
                print('trying to get pdb file')
                urllib.request.urlretrieve('https://files.rcsb.org/download/{}.pdb'.format(id), path)
                break
            except IndexError:
                print('failed getting pdb file')
                if fails == 10:
                    return False
                fails += 1
    return True


def read_pdb_entry(pdb_id):
    file = pdb_id + '.txt'
    path = 'split/{}'.format(file)
    if not os.path.isfile(path):
        print('\tFile not found: {}'.format(path))
        return None
    with open(path, 'r') as pdb_txt_file:
        ret = {}
        ret['id'] = file[-10:-6]
        ret['chain'] = file[-5]
        ret['sequence'] = pdb_txt_file.readline().strip()
        ret['secstr'] = pdb_txt_file.readline().strip()
        ret['disorder'] = pdb_txt_file.readline().strip()
    return ret


def pdbid2go(pdbid, chain):
    path = 'xml_files/{}.xml'.format(pdbid)
    fails = 0
    if not os.path.exists(path):
        while True:
            try:
                print('Trying to download go terms')
                urllib.request.urlretrieve('https://www.rcsb.org/pdb/rest/goTerms?structureId={}'.format(pdbid),
                                           path)
                break
            except NotImplementedError:
                print('Failed to download the go terms')
                if fails == 10:
                    return False
                fails += 1
    gos = []
    with open(path, 'r') as ifile:
        for line in ifile:
            if '<term id=' in line:  # <term id="GO:0020037" structureId="4HHB" chainId="D">
                line = line.split('"')
                if line[5] == chain:
                    gos.append(line[1])
    return gos


def get_pfam_file(pfam_id, with_up=False):
    print('Seaching for pfam for {}.'.format(pfam_id))
    path = 'pfam_files/{}.pfam'.format(pfam_id)
    if not os.path.exists(path):
        fails = 0
        while True:
            try:
                print('Trying to get seed Pfam')
                urllib.request.urlretrieve('https://pfam.xfam.org/family/{}/alignment/seed'.format(pfam_id), path)
                break
            except IndexError:
                print('Failed to get seed Pfam')
                if fails == 10:
                    return False
                fails += 1

    return True


def get_pdb_id_from_pfam_alignment(pfam_id):
    path = 'pfam_files/{}.pfam'.format(pfam_id)
    with open(path, 'r') as ifile:
        for line in ifile:
            if 'DR' in line:
                if 'PDB' in line:
                    return line.split(';')[1].strip().replace(' ', '_').upper()


def parse_pdb(number_of_entries, priorities):
    """parses pdb file downloaded from https://cdn.rcsb.org/etl/kabschSander/ss_dis.txt.gz and makes masked datasets"""
    number_of_entries = int(number_of_entries)
    if priorities == '':
        priorities = []
    else:
        priorities = priorities.split(',')
    print('priorities are: {}'.format(priorities))
    pdb2pfam = {}
    with open('pdb_chain_pfam.shuffled.csv', 'r') as ifile:
        ifile.readline()
        for line in ifile:
            line = line.strip().split(',')

            pdb_chain = line[0].upper() + '_' + line[1].upper()
            pdb2pfam[pdb_chain] = line[3]

    print('Got pdb_chain_pfam.csv')

    pdb_ids = list(pdb2pfam.keys())[:2 * number_of_entries]
    random.shuffle(pdb_ids)

    print('Processing {} entries.'.format(number_of_entries))
    os.makedirs('sensitivity_ds', exist_ok=True)

    masked_dataset_file = open('masked_dataset.txt', 'w')
    # check priority stuff
    for pdb_id in priorities:
        current_entry = read_pdb_entry(pdb_id)
        if not current_entry:
            print('\tCould not find ss_dis entry for {}.'.format(pdb_id))
            continue

        gos = pdbid2go(pdb_id[:4], pdb_id[5])
        if not len(gos) > 0:
            print('\tNo gos found for {}.'.format(pdb_id))
            continue

        success = get_pdb_file(pdb_id[:4])
        if not success:
            print('\tCould not get pdb file for {}.'.format(pdb_id))

        success = get_pfam_file(pdb2pfam[pdb_id], with_up=True)
        if not success:
            print('\tCould not get pfam file for {}.'.format(pdb_id))

        masked_dataset_file.write('{}\n'.format(';'.join([current_entry['id'],
                                                          current_entry['chain'],
                                                          ','.join(gos),
                                                          current_entry['sequence'],
                                                          current_entry['secstr'],
                                                          current_entry['disorder']])))
        masked_dataset_file.flush()
        print('Wrote {}.'.format(pdb_id))

    # non-priorities
    count = 0
    for pfam_id in pdb2pfam.values():
        success = get_pfam_file(pfam_id, with_up=False)
        if not success:
            print('\tCould not get pfam file for {}.'.format(pfam_id))
        pdb_id = get_pdb_id_from_pfam_alignment(pfam_id)
        if not pdb_id:
            continue

        current_entry = read_pdb_entry(pdb_id)
        if not current_entry:
            print('\tCould not find ss_dis entry for {}.'.format(pdb_id))
            continue

        gos = pdbid2go(pdb_id[:4], pdb_id[5])
        if not len(gos) > 0:
            print('\tNo gos found for {}.'.format(pdb_id))
            continue

        success = get_pdb_file(pdb_id[:4])
        if not success:
            print('\tCould not get pdb file for {}.'.format(pdb_id))

        masked_dataset_file.write('{}\n'.format(';'.join([current_entry['id'],
                                                          current_entry['chain'],
                                                          ','.join(gos),
                                                          current_entry['sequence'],
                                                          current_entry['secstr'],
                                                          current_entry['disorder']])))
        masked_dataset_file.flush()
        count += 1
        print('Wrote {}.'.format(pdb_id))
        if count == number_of_entries:
            break
    masked_dataset_file.close()
    print('Done')
    return


if __name__ == '__main__':
    number_of_entries = sys.argv[1]
    try:
        priorities = sys.argv[2]
    except:
        priorities = ''

    parse_pdb(number_of_entries, priorities)
