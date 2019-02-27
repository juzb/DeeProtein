#!/usr/bin/env bash
# assumes the current working directory is the data directory, leaves in the starting directory
CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3


# write the cafa3 datasets:
# 1. test_cafa3.csv             test dataset, annotated by current swissprot
# 2. train_cafa3_original.csv   train dataset, using only the annotations provided by cafa3 (as downloaded)
# 3. train_cafa3_expanded.csv   train dataset, using the sequences as provided by cafa3, but the current swissprot annotations

#
mkdir -p processed_sp
cd processed_sp
#
## write sp csv
python "$CODE_DIR"/DeeProtein/datapreprocessing/datapreprocessing.py . ../raw_data/uniprot_sprot.dat "$GO_DAG"
sort -k1 -t';' filtered_uniprot.csv > filtered_sp.sorted.csv # daatapreprocessing writes filtered_uniprot.csv
#
echo 'Number of sequences before filtering: ' >> sp.log
grep -c '//' ../raw_data/uniprot_sprot.dat >> sp.log
echo '\nNumber of sequences after filtering: ' >> sp.log
wc -l filtered_sp.sorted.csv >> sp.log
#
cd ../
#
## cafa stuff
mkdir -p processed_cafa3
cd processed_cafa3
#
### 1. test 
python "$CODE_DIR"/DeeProtein/scripts/cafa_targets_parser.py ../raw_cafa3 test_cafa3.csv "$GO_DAG" ../processed_sp/filtered_sp.sorted.csv
shuf test_cafa3.csv > test_cafa3.shuffled.csv
# Create a stats file
python "$CODE_DIR"/DeeProtein/scripts/calculate_dataset_stats.py test_cafa3.csv stats.test_cafa3.csv

# build a 100 entry version for unit tests
head -n 100 test_cafa3.shuffled.csv > test_cafa3.unittest.csv
# Create a stats file
python "$CODE_DIR"/DeeProtein/scripts/calculate_dataset_stats.py test_cafa3.unittest.csv stats.test_cafa3.unittest.csv

## 1. test reduced for deepgo comparison 
python "$CODE_DIR"/DeeProtein/scripts/extract_targets.py ../comparison_data/deepgo_eval_data/deepgo/model_preds_filtered_mf.pkl ../raw_cafa3 test_cafa3_deepgo_comparison.csv deepgo_comparison_go_file.txt
shuf test_cafa3_deepgo_comparison.csv > test_cafa3_deepgo_comparison.shuffled.csv
# Create a stats file
python "$CODE_DIR"/DeeProtein/scripts/calculate_dataset_stats.py test_cafa3_deepgo_comparison.csv stats.test_cafa3_deepgo_comparison.csv
echo "1. test reduced for deepgo comparison" >> /dev/stderr

## 2. train original 
python "$CODE_DIR"/DeeProtein/scripts/cafa_training_parser.py ../raw_cafa3/CAFA3_training_data/uniprot_sprot_exp train_cafa3_original.csv deepgo_comparison_go_file.txt false "$GO_DAG"
shuf train_cafa3_original.csv > train_cafa3_original.shuffled.csv
# Create a stats file
python "$CODE_DIR"/DeeProtein/scripts/calculate_dataset_stats.py train_cafa3_original.csv stats.train_cafa3_original.csv
echo "2. train original" >> /dev/stderr
#

#
cd ../
#
cd processed_sp
## chits stuff
## write fastas
python "$CODE_DIR"/DeeProtein/scripts/csv2fasta.py filtered_uniprot.csv filtered_sp.fasta # Beware with filtered_uniprot naming
python "$CODE_DIR"/DeeProtein/scripts/csv2fasta.py ../processed_cafa3/test_cafa3.shuffled.csv ../processed_cafa3/test_cafa3.fasta
#
#
## 1.cdhits
## 2.convert back
## 3.shuf
CAFA_FASTA='../processed_cafa3/test_cafa3.fasta'
#
IDENTITY='0.5'
WORDSIZE='3'
cd-hit-2d -i $CAFA_FASTA -i2 filtered_sp.fasta -o cdhitted_sp_05.fasta -c $IDENTITY -s2 $IDENTITY -n $WORDSIZE -T 0 -M 0 -d 0
python "$CODE_DIR"/DeeProtein/scripts/prune_on_cdhit.py cdhitted_sp_05.fasta filtered_uniprot.csv filtered_sp_cdhitted_05.csv
shuf filtered_sp_cdhitted_05.csv > filtered_sp_cdhitted_05.csv.shuffled
# Create a stats file
python "$CODE_DIR"/DeeProtein/scripts/calculate_dataset_stats.py filtered_sp_cdhitted_05.csv stats.filtered_sp_cdhitted_05.csv
wc -l filtered_sp_cdhitted_05.csv >> cdhitted.log
cd ../

# 5. sensitivity dataset
cd rcsb
mkdir -p split
python "$CODE_DIR"/DeeProtein/scripts/split_ss_dis_file.py ss_dis.txt split/

python "$CODE_DIR"/DeeProtein/scripts/sensitivity_ds.py 2000 1ERK_A,1GOL_A,5VW1_B,4UN3_B,6F3F_A,1X86_B,1CEE_A,3QBV_B,3ODO_A,2DX1_A,1EJE_A,4VHB_A > sensitivity_ds.log

cat ../instructions/masked_dataset_manually.txt masked_dataset.txt > extended_masked_dataset.txt

cd ../

echo "datasets.sh DONE."
