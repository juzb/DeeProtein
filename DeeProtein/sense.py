import subprocess

while True:
    name =      input('Please enter four letter name for this run: ')
    sequence =  input('Please enter the sequence to analyse: ')
    gos =       input('Please enter the GO terms to analyse sperarated by commas: ')
    
    with open('/data/data/masked_datasets/masked_dataset.txt', 'w') as ofile:
        ofile.write('{};{};{};{};{};{}'.format(name,
                                               'A',
                                               gos,
                                               sequence,
                                               '.' * len(sequence),
                                               '_' * len(sequence)))
    subprocess.call(['bash', 'analyze_sensitivity.sh'])
    print('Performed sensitivity analysis. Please find the results in /output/out/aa_resolution/masked_{}_A_1.txt\n\n'.format(name))