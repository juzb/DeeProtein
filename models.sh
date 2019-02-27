#!/usr/bin/env bash

# assumes the current working directory is the data directory, leaves in the starting directory
CODE_DIR=$1
GO_LIST=$2
GO_DAG=$3

mkdir -p models
cd models


# deeprotein training on cafa3 train and on swissprot train
for ds in  "sp_" #"up_" "ca_"
do
    if [ "$ds" == "sp_" ]
        then
        ne="20"
        td="../processed_sp/filtered_sp_cdhitted_05.csv.shuffled"
    elif [ "$ds" == "ca_" ]
        then
        ne="20"
        td="../processed_cafa3/train_cafa3_original.shuffled.csv"
    elif [ "$ds" == "up_" ]
        then
        ne="1"
        td="../processed_up/filtered_up_cdhitted_05.csv.shuffled"
    else
        echo "Something went wrong with $ds" >> /dev/stderr
        exit 1 # something went horribly wrong
    fi

    for idx in 0 1 2 3
        do
        CUDA_VISIBLE_DEVICES="$idx" python "$CODE_DIR"/DeeProtein/DeeProtein.py -n "$ds$idx" -dp true -ne "$ne" -t "$td" -v ../processed_cafa3/test_cafa3_deepgo_comparison.shuffled.csv -g "$GO_LIST" -ip "$ds$idx" -mo "$CODE_DIR"/DeeProtein/model/model.py -b 64 -va 100000 -gof "$GO_DAG" -r False -rl False -m False -i False -c False -te False &
        echo "DeeProtein training $ds$idx" >> /dev/stderr
    done
    wait
done

cd ../
