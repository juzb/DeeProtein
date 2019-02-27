#!/usr/bin/env bash
# assumes the current working directory is the data directory, leaves in the starting directory

# Downloads swissprot and uniprot

mkdir -p raw_data
cd raw_data

#wget ftp://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz &
wget ftp://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz &
wait
#gunzip uniprot_trembl.dat.gz &
gunzip uniprot_sprot.dat.gz &
wait
#rm uniprot_trembl.dat.gz
rm uniprot_sprot.dat.gz
cd ../

# original cafa3 data
mkdir -p raw_cafa3
cd raw_cafa3

# download the cafa3 data
wget https://biofunctionprediction.org/cafa-targets/CAFA3_targets.tgz
wget https://biofunctionprediction.org/cafa-targets/CAFA3_training_data.tgz

tar -xzf CAFA3_targets.tgz
tar -xzf CAFA3_training_data.tgz

cd ../


# get sequences of PDB files:
mkdir -p rcsb
cd rcsb
wget https://cdn.rcsb.org/etl/kabschSander/ss_dis.txt.gz
gunzip ss_dis.txt.gz

wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_pfam.csv.gz
gunzip pdb_chain_pfam.csv
tail -n +2 pdb_chain_pfam.csv | shuf > pdb_chain_pfam.shuffled.csv

wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_interpro.csv.gz
gunzip pdb_chain_interpro.csv


mkdir -p xml_files
mkdir -p pdb_files
mkdir -p pfam_files
cd ../



# download the data to compare against
mkdir -p comparison_data
cd comparison_data

# DeepGO data, also containing predictions made by gofdr

wget https://github.com/bio-ontology-research-group/deepgo/raw/master/eval_data.tar.gz
tar -xzf eval_data.tar.gz
mv data deepgo_eval_data
mv eval_data.tar.gz deepgo_eval_data/

## the files in comparison_data/deepgo_eval_data/deepgo/ are readable with pandas.read_pickle()

## mf.csv contains a list of the molecular function go terms evaluated
python -c "import pandas;pandas.read_pickle('deepgo_eval_data/deepgo/mf.pkl').to_csv('deepgo_eval_data/deepgo/mf.csv')"

cd ../



echo "downloads.sh DONE."