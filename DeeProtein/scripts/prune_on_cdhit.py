import sys

with open(sys.argv[1], "r") as fasta, open(sys.argv[2], "r") as incsv, open(sys.argv[3], "w") as outcsv:
    fasta_set = {line[1:-1] for line in fasta if line[0] == ">"}
    for line in incsv:
        id = line.split(";")[0]
        if id in fasta_set:
            outcsv.write(line)
        else:
            pass
