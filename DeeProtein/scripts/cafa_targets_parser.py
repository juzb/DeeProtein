import sys
import os
import urllib.request
import http
from goatools.obo_parser import GODag


def parse(path, out_path, godag_file, sp_csv):
    print('Parsing {}'.format(path))

    problem_path = 'test_problems.csv'

    target_path = os.path.join(path, 'Target files')
    map_path = os.path.join(path, 'Mapping files')
    godag = GODag(godag_file,
                  optional_attrs=['relationship'])

    tries = 10

    spid2gos = {}
    with open(sp_csv, 'r') as ifile:
        for line in ifile:
            line = line.split(';')
            spid2gos[line[0]] = line[2].split(',')
    print('Got spid2gos dict with {} elements.'.format(len(spid2gos)))

    file_numbers = []
    map_files = os.listdir(map_path)
    sp_map_files = [f for f in map_files if 'sp_species' in f]
    for file in sp_map_files:
        file_numbers.append(file.split('.')[1])

    print('Found {} files in swissprot list.'.format(len(file_numbers)))

    print('Writing to {} from {}'.format(out_path, os.getcwd()))
    # get output file
    out_file = open(out_path, 'w')
    problems_file = open(problem_path, 'w')

    successes = 0
    problems = 0
    # get target file and mapping file
    for file_number in file_numbers:
        print('Working with file {}'.format(file_number))
        target_file = os.path.join(target_path, 'target.{}.fasta'.format(file_number))

        # retrieve gos from uniprot
        seq = ""
        sp_id = ""
        gos = []
        with open(target_file, 'r') as tf:  # , open(map_file, 'r') as mf:
            for line in tf:
                if not line.startswith('>'):
                    seq += line.strip()
                    if len(seq) > 1000:
                        seq = ""
                        sp_id = ""
                        continue
                else:
                    old_sp_id = sp_id
                    sp_id = line.split()[1].strip()

                    if len(old_sp_id) > 0:
                        parents = []
                        for go in gos:
                            try:
                                parents += godag[go].get_all_parents()
                            except KeyError:
                                pass
                        gos += parents

                        gos = list(set(gos))  # remove duplicates
                        if len(gos) > 0:
                            out_file.write('{};{};{}\n'.format(old_sp_id, seq, ','.join(gos)))
                            successes += 1
                        else:
                            problems += 1
                    if (successes + problems) % 100 == 0:
                        print('{} successes and {} problems so far.'.format(successes, problems))
                        problems_file.flush()
                        out_file.flush()

                    seq = ""
                    try:
                        gos = spid2gos[sp_id]
                    except KeyError:
                        problems_file.write('{}\n'.format(sp_id))
                        problems += 1
                        seq = ""
                        sp_id = ""
                        continue

    print('Wrote to {}'.format(out_path))

    # close files
    out_file.close()
    problems_file.close()
    print('cafa targets parser DONE.')


if __name__ == '__main__':
    path = sys.argv[1]
    out_path = sys.argv[2]
    godag_file = sys.argv[3]
    sp_csv = sys.argv[4]

    parse(path, out_path, godag_file, sp_csv)
