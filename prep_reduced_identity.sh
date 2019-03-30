#!/usr/bin/env bash

# assumes the current working directory is the data directory, leaves in the starting directory

CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3

set -ex # Verbose STDERR and  break on fail

mkdir -p eval_ri
cd eval_ri

# Create blast database
makeblastdb -in ../processed_sp/cdhitted_sp_05.fasta -out sp_train_db -parse_seqids
mkdir blast_result_dir
cd blast_result_dir
blastp -num_threads 112 -db ../sp_train_db -query ../../processed_cafa3/test_cafa3.fasta -outfmt "7 qacc sacc pident nident slen" | csplit -f blast_result_ -n 9 - /BLASTP/ {*} -z
cd ..
python $CODE_DIR/DeeProtein/scripts/reduce_identity.py blast_result_dir ../processed_cafa3/test_cafa3.csv test_ri

cd ../
