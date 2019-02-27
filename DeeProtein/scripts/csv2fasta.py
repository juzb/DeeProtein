import sys

with open(sys.argv[1], 'r') as ifile, open(sys.argv[2], 'w') as ofile:
    for line in ifile:
        line = line.strip().split(';')
        ofile.write('>{}\n{}\n'.format(line[0], line[1]))
print('Done')
