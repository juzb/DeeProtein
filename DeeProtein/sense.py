import subprocess

while True:
    #name = input('Please enter four letter name for this run: ')
    name = "AAAA"
    sequence = input('Please enter the sequence to analyze: ')
    gos = input('Please enter the GO terms to analyze sperarated by commas: ')

    with open('/results/tmp/masked_dataset.txt', 'w') as ofile:
        ofile.write('{};{};{};{};{};{}'.format(name,
                                               'A',
                                               gos,
                                               sequence,
                                               '.' * len(sequence),
                                               '_' * len(sequence)))

    subprocess.call(['bash', '/code/analyze_sensitivity.sh', gos])
    print('Performed sensitivity analysis. '
          'Please find the results in /results\n\n')

