#!/bin/bash

export PATH="/opt/conda/lib:$PATH"

echo "Sequence" $1;
echo "GOs"      $2;
echo "Task"     $3;

if [ -z "$1" ]; then
    SEQ="GARMVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMNKGSLLDFLKGETGKYLRLPQLVDMSAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL";
    echo $SEQ
else
    SEQ=$1
fi

if [ -z "$2" ]; then
    GO="GO:0016301,GO:0005524";
    echo $GO
else
    GO=$2
fi

if [ -z "$3" ]; then
    MODE="default";
    echo $MODE
else
    MODE=$3
fi

if [ "$MODE" = "C" ] || [ "$MODE" = "Classification" ]; then
    echo "Classification of $SEQ";
    python -u /code/DeeProtein/DeeProtein.py -restore=True -sq $SEQ -v /results/;

elif [ "$MODE" = "default" ]; then
    echo "Performing Classification and Sensitivity Analysis for $SEQ and $GO";
    python -u /code/DeeProtein/DeeProtein.py -restore=True -sq $SEQ -v /results/;
    
    len=${#SEQ};
    
    echo "" >> /output/out/results.txt
    echo "" >> /output/out/results.txt
    echo "Sensitivity analysis" >> /output/out/results.txt
    echo "--------------------" >> /output/out/results.txt
    
    
    points=$(printf '.%.0s' $(eval echo "{1..$len}"));
    scores=$(printf '_%.0s' $(eval echo "{1..$len}"));
    echo "AAAA;A;$GO;$SEQ;$points;$scores" > "/results/masked_dataset.txt";
    python -u /code/DeeProtein/DeeProtein.py -mask=True -restore=True -v /results/masked_dataset.txt;
    python -u /code/DeeProtein/scripts/calculate_sensitivity.py /output/out/aa_resolution/masked_AAAA_A_1.txt /output/out/aa_resolution/masked_AAAA_A_1.txt;
    cat /output/out/aa_resolution/masked_AAAA_A_1.txt >> /output/out/results.txt;

    python -u /code/DeeProtein/scripts/plot_sensitivity.py /output/out/aa_resolution/masked_AAAA_A_1.txt "$GO" /output/out/AAAA_A_1.png 


else
    echo "Sensitivity Analysis of $SEQ";
    echo "for $GO";
    len=${#SEQ};

    points=$(printf '.%.0s' $(eval echo "{1..$len}"));
    scores=$(printf '_%.0s' $(eval echo "{1..$len}"));  
    echo "AAAA;A;$GO;$SEQ;$points;$scores" > "/results/masked_dataset.txt";
    
    python -u /code/DeeProtein/DeeProtein.py -mask=True -restore=True -v /results/masked_dataset.txt;
    python -u /code/DeeProtein/scripts/calculate_sensitivity.py /output/out/aa_resolution/masked_AAAA_A_1.txt /output/out/aa_resolution/masked_AAAA_A_1.txt;
    
    cp /output/out/aa_resolution/masked_AAAA_A_1.txt /output/out/results.txt;
    
    python -u /code/DeeProtein/scripts/plot_sensitivity.py /output/out/aa_resolution/masked_AAAA_A_1.txt "$GO" /output/out/AAAA_A_1.png 

fi

echo "Done";
