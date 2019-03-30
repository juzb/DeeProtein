#!/usr/bin/env bash

# assumes the current working directory is the data directory, leaves in the starting directory

CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3


resolve_stats(){ # For evaluation the proper dataset stat csv file has to be loaded
if [ "$1" == "sp_" ]
    then
    STAT_FILE="../processed_sp/stats.filtered_sp_cdhitted_05.csv"
elif [ "$1" == "up_" ]
    then
    STAT_FILE="../processed_up/stats.filtered_up_cdhitted_05.csv"
elif [ "$1" == "ca_" ]
    then
    STAT_FILE="../processed_cafa3/stats.train_cafa3_original.csv"
else
    echo "Unknown dataset prefix"
    exit 1
fi
}

set -ex # Verbose STDERR and  break on fail

mkdir -p eval_ri
cd eval_ri


ds="sp_"

for es in $(ls test_ri);
do
    resolve_stats "$ds"
    for idx in 0 1 2 3
        do
        CUDA_VISIBLE_DEVICES=$idx python "$CODE_DIR"/DeeProtein/DeeProtein.py -n $es"_"$idx -dp true -ne 1 -v "test_ri/"$es -ds $STAT_FILE -g "$GO_LIST" -ip "full_"$es"_"$idx -mo "$CODE_DIR"/DeeProtein/model/model.py -b 64 -gof "$GO_DAG" -r True -rl False -m False -i False -te True -rp "../models/$ds$idx/saves/" &
    done
    wait
done

cd ../
