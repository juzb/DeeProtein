DATA_DIR="" # Specify your directory for the data here
CODE_DIR="" # Specify the absolute path to this code directory
GO_LIST="${CODE_DIR}/instructions/go_file.txt" # File specifiying the go terms considered (Change if necessary)
GO_DAG="" # Specify the path to the .obo file of the gene ontology

# Options
set -e # Stop execution on error
# set -x # Print all calls for debugging

# make the directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
mkdir -p logs

# download the needed files
bash "$CODE_DIR"/downloads.sh
echo "DONE with downloads" >> /dev/stderr

# write the datasets
bash "$CODE_DIR"/datasets.sh $CODE_DIR $GO_LIST $GO_DAG

# Attention the following command is very CPU and memory intensive and might take very long
# Uncomment if necessary

#bash "$CODE_DIR"/datasets_up.sh $CODE_DIR $GO_LIST $GO_DAG # Processing UniProt

echo "DONE with datasets" >> /dev/stderr

# BEGIN SECTION requiring GPU

# train the models
bash "$CODE_DIR"/models.sh $CODE_DIR $GO_LIST $GO_DAG
echo "DONE with models" >> /dev/stderr

# run the evaluation
bash "$CODE_DIR"/eval.sh $CODE_DIR $GO_LIST $GO_DAG
echo "DONE with evaluation" >> /dev/stderr

# run the sensitivity analysis
bash "$CODE_DIR"/sensitivity.sh $CODE_DIR $GO_LIST $GO_DAG
echo "DONE with sensitivity" >> /dev/stderr

# END SECTION requiring GPU

# run the analysis
bash "$CODE_DIR"/analysis.sh $CODE_DIR $GO_LIST $GO_DAG
echo "DONE with analysis" >> /dev/stderr

echo "DONE."

