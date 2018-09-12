import os
import sys

"""
Reads the file from https://cdn.rcsb.org/etl/kabschSander/ss_dis.txt.gz (sys.argv[1]) 
and splits it to have one file for each protein
Replaces all spaces with dots so that these don't interfere with split()
writes to second sys.argv
"""

path = sys.argv[2]

first = True

with open(sys.argv[1], 'r') as ifile:
    for line in ifile:
        line = line.replace(' ', '.').strip()
        if line.startswith('>'):
            if line.endswith('sequence'): # new entry:
                pdbid = line[1:5]
                chain = line[6]
                try:
                    ofile.close()
                except:
                    pass
                ofile = open(os.path.join(path, '{}_{}.txt'.format(pdbid, chain)), 'w')
            else:
                ofile.write('\n')
        else:
            ofile.write(line)

print('DONE.')
