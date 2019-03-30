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

mkdir -p eval
cd eval

# evaluate the performance of deepgo: this is done once based on data downloaded from the papers supplements
python "$CODE_DIR"/DeeProtein/scripts/deep_go_eval.py ../comparison_data/deepgo_eval_data/deepgo/model_preds_filtered_mf.pkl "$GO_DAG"  ../comparison_data/deepgo_eval_data/deepgo/mf.pkl > deep_go_eval.txt
echo "evaluate the performance of deepgo" >> /dev/stderr

# deeprotein training on cafa3 train


# 1 deeprotein testing for comparison against deepgo
for ds in "sp_" # "ca_" "up_"
do
    resolve_stats "$ds"
    for idx in 0 1 2 3
        do
        CUDA_VISIBLE_DEVICES=$idx python "$CODE_DIR"/DeeProtein/DeeProtein.py -n "$ds$idx" -dp true -ne 1 -v ../processed_cafa3/test_cafa3_deepgo_comparison.shuffled.csv -ds $STAT_FILE -g "$GO_LIST" -ip "benchmark_$ds$idx" -mo "$CODE_DIR"/DeeProtein/model/model.py -b 64 -gof "$GO_DAG" -r True -rl False -m False -i False  -te True -rp "../models/$ds$idx/saves/" &
        echo "deeprotein testing for comparison against deepgo" >> /dev/stderr
    done
    wait
done


# 2 deeprotein testing on full cafa3 test set
for ds in "sp_" # "ca_" "up_"
do
    resolve_stats "$ds"
    for idx in 0 1 2 3
        do
        CUDA_VISIBLE_DEVICES=$idx python "$CODE_DIR"/DeeProtein/DeeProtein.py -n "$ds$idx" -dp true -ne 1 -v ../processed_cafa3/test_cafa3.shuffled.csv -ds $STAT_FILE -g "$GO_LIST" -ip "full_$ds$idx" -mo "$CODE_DIR"/DeeProtein/model/model.py -b 64 -gof "$GO_DAG" -r True -rl False -m False -i False  -te True -rp "../models/$ds$idx/saves/" &
        echo "deeprotein testing on full cafa3 test set" >> /dev/stderr
    done
    wait
done


for ds in "sp_" # "ca_" "up_"
do
    for idx in 0 1 2 3
        do
        # 3 parse results to fit to deepgo cafa3 evaluation dataframe
        python "$CODE_DIR"/DeeProtein/scripts/convert_test_predictions.py "benchmark_$ds$idx/" ../comparison_data/deepgo_eval_data/deepgo/model_preds_filtered_mf.pkl
        echo "parse results to fit to deepgo cafa3 evaluation dataframe" >> /dev/stderr

        # 4 evaluate the performance of DeeProtein:
        python "$CODE_DIR"/DeeProtein/scripts/deep_go_eval.py "benchmark_$ds$idx/metrics/dg_like_df.pkl" "$GO_DAG"  ../comparison_data/deepgo_eval_data/deepgo/mf.pkl > "benchmark_$ds$idx.txt"
        echo "evaluate the performance of DeeProtein" >> /dev/stderr
    done
done


cd ../
