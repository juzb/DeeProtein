#!/usr/bin/env bash

# assumes the current working directory is the data directory, leaves in the starting directory
CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3

for ds in "sp_" #"ca_" "up_" 
    do
    # Calculate distances to ligands
    mkdir -p "$ds"ligand_binding

    python "$CODE_DIR"/DeeProtein/scripts/distance.py  sensitivity/"$ds"combined_mean/ rcsb/pdb_files/ instructions/ligand_binding.csv "$ds"ligand_binding/ 5 "$CODE_DIR"/DeeProtein/style/

    mkdir -p "$ds"active_site

    python "$CODE_DIR"/DeeProtein/scripts/distance.py sensitivity/"$ds"combined_mean/ rcsb/pdb_files/ instructions/active_site.csv "$ds"active_site/ 5 "$CODE_DIR"/DeeProtein/style/

   # fuse with information content
    python "$CODE_DIR"/DeeProtein/scripts/add_ic.py sensitivity/"$ds"combined_full/ rcsb/pfam_files/ rcsb/pdb_chain_pfam.csv
    # calculate all the correlations
    mkdir -p sensitivity/"$ds"correlations
    python "$CODE_DIR"/DeeProtein/scripts/correlator.py sensitivity/"$ds"combined_full/ sensitivity/"$ds"combined_full_ic/ $GO_DAG $GO_LIST sensitivity/"$ds"correlations
    done

echo "analysis DONE."
