#!/usr/bin/env bash
# assumes the current working directory is the data directory, leaves in the starting directory
CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3


# modification of datasets.sh for uniprot datasets only

mkdir -p processed_up
cd processed_up

## write sp csv
python "$CODE_DIR"/DeeProtein/datapreprocessing/datapreprocessing.py . ../raw_data/uniprot_sprot.dat "$GO_DAG"
#
echo 'Number of sequences before filtering: ' >> up.log
grep -c '//' ../raw_data/uniprot_trembl.dat >> up.log
echo '\nNumber of sequences after filtering: ' >> up.log
wc -l filtered_uniprot.csv >> up.log
#
# Reduce Uniprot/TrEMBL dataset to 20 Mio entries to reduce computational cost
shuf filtered_uniprot.csv > filtered_uniprot.shuf.csv
head -n 20000000 filtered_uniprot.csv > reduced_uniprot.csv


## cdhits stuff
## write fastas
python "$CODE_DIR"/DeeProtein/scripts/csv2fasta.py reduced_uniprot.csv reduced_up.fasta 
#
#
##cdhits
CAFA_FASTA='../processed_cafa3/test_cafa3.fasta'
#
IDENTITY='0.5'
WORDSIZE='3'
cd-hit-2d -i $CAFA_FASTA -i2 reduced_up.fasta -o cdhitted_up_05.fasta -c $IDENTITY -s2 $IDENTITY -n $WORDSIZE -T 0 -M 0 -d 0
python "$CODE_DIR"/DeeProtein/scripts/prune_on_cdhit.py cdhitted_up_05.fasta reduced_uniprot.csv filtered_up_cdhitted_05.csv
# Create a stats file
python "$CODE_DIR"/DeeProtein/scripts/calculate_dataset_stats.py filtered_up_cdhitted_05.csv stats.filtered_up_cdhitted_05.csv
wc -l filtered_up_cdhitted_05.csv >> cdhitted.log
#

cd ../

echo "datasets.sh DONE."
