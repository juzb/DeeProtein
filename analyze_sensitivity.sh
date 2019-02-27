#!/bin/bash

echo "Running DeeProtein.py for sensitivity analysis..."

# run the sensitivity analysis
python -u /code/DeeProtein/DeeProtein.py -mask=True -restore=True -v /results/tmp/masked_dataset.txt -ip /results/tmp/

# process the raw values
python -u /code/DeeProtein/scripts/calculate_sensitivity.py /results/tmp/aa_resolution/masked_AAAA_A_1.txt /results/sensitivities.txt

# plot the processed values ($1 holds a comma-separated list of GO terms)
python -u /code/DeeProtein/scripts/plot_sensitivity.py /results/sensitivities.txt "$1" /results/sensitivities.png 

