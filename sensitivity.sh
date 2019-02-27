#!/usr/bin/env bash

# assumes the current working directory is the data directory, leaves in the starting directory
CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3

mkdir -p sensitivity
cd sensitivity

for ds in  "sp_" # "up_" "ca_" 
    do
    for idx in 0 1 2 3
        do
        CUDA_VISIBLE_DEVICES="$idx" python "$CODE_DIR"/DeeProtein/DeeProtein.py -ip "$ds$idx" -n "$ds$idx" -mo "$CODE_DIR"/DeeProtein/model/model.py -gof "$GO_DAG" -rl True -m True -sas false -ne 20 -v ../rcsb/extended_masked_dataset.txt -g "$GO_LIST" -mw 1 -b 1 -rp "../models/$ds$idx/saves/" &
        echo "Sensitivity $ds$idx" >> /dev/stderr
    done
    wait
    echo "Sensitivity done." >> /dev/stderr

    for idx in 0 1 2 3
        do
        python "$CODE_DIR"/DeeProtein/scripts/calculate_sensitivity.py $ds$idx/aa_resolution $ds$idx/aa_resolution_c &
        echo "Sensitivity II $ds$idx" >> /dev/stderr
    done
    wait

    # combine
    python "$CODE_DIR"/DeeProtein/scripts/combine_masked_dumps.py . "$ds" aa_resolution_c
    
done
    

cd ../
echo "sensitivity DONE."

