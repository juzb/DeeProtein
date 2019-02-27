import pandas as pd
import sys
import os

deep_go_data = sys.argv[1]
dataset_path_in = sys.argv[2]
dataset_path_out = sys.argv[3]
go_file_out = sys.argv[4]

df = pd.read_pickle(deep_go_data)

targets = list(df.targets)
gos = list(df.gos)

go_freqs = {}
target2go = dict(zip(targets, gos))

target_path = os.path.join(dataset_path_in, 'Target files')
map_path = os.path.join(dataset_path_in, 'Mapping files')

file_numbers = []
map_files = os.listdir(map_path)
sp_map_files = [f for f in map_files if 'sp_species' in f]
for file in sp_map_files:
    file_numbers.append(file.split('.')[1])

with open(dataset_path_out, 'w') as ofile:
    for file_number in file_numbers:
        print('Working with file {}'.format(file_number))
        target_file = os.path.join(target_path, 'target.{}.fasta'.format(file_number))

        # retrieve gos from uniprot
        seq = ""
        saving = False
        with open(target_file, 'r') as ifile:
            for line in ifile:
                if line.startswith('>'):
                    if saving:
                        saving = False
                        gos = target2go[target_id]
                        for go in gos:
                            if not go in go_freqs:
                                go_freqs[go] = 1
                            else:
                                go_freqs[go] += 1
                        if not len(seq) > 1000:
                            if not len(gos) == 0:
                                ofile.write('{};{};{}\n'.format(target_id, seq, ','.join(gos)))
                            else:
                                print('No gos found for target {}'.format(target_id))
                        else:
                            print('Sequence too long for target {}'.format(target_id))
                        seq = ""
                    target_id = line[1:].split()[0].strip()
                    if target_id in target2go:
                        print('OK for tareget {}'.format(target_id))
                        targets.remove(target_id)
                        saving = True
                    else:
                        saving = False
                else:
                    if saving:
                        seq += line.strip()

print('Did not find entries for:\n{}'.format('\n'.join(targets)))

go_freq_keys = sorted(list(go_freqs.keys()), key=lambda x: -go_freqs[x])
with open(go_file_out, 'w') as ofile:
    for go in go_freq_keys:
        ofile.write(' {} {}.csv\n'.format(go_freqs[go], go))
