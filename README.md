# DeeProtein
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2574979.svg)](https://doi.org/10.5281/zenodo.2574979)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2577920.svg)](https://doi.org/10.5281/zenodo.2577920)


[Github repository](https://github.com/juzb/DeeProtein)

[Code ocean compute capsule]()

Software for "Leveraging Implicit Knowledge in Neural Networks for Functional Dissection and Engineering of Proteins"


# Online Mode

With just a few clicks, you can run DeeProtein online in our code ocean compute capsule. To do so, you need to sign up and duplicate this capsule.

This compute capsule includes two modes:

* **Classification:** predict the functions of any protein sequence
* **Sensitivity analysis:** run a full sensitivity analysis for a set of GO terms for any protein sequence

For both modes click *Start Interactive Session* choose *Juypter*.
In the dropdown menu *New* on the righthand side choose *Terminal*. 

## Classification

Wait until the terminal opens and enter

```bash
bash /code/infer.sh
```

to start the inference mode. 

**Example:**

A prompt will open where you can enter a protein sequence, e.g the sequence of Src-kinase kinase ([PDB 6F3F](https://www.rcsb.org/structure/6f3f))

```bash
Enter Sequence for Classification:

GARMVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMNKGSLLDFLKGETGKYLRLPQLVDMSAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL
```

For which the result looks like this:

```txt
Predicted labels:
-----------------

GO-Term         Score   Explanation
GO:0003674      1.000   molecular_function
GO:0043168      1.000   anion binding
GO:0004672      1.000   protein kinase activity
GO:0035639      1.000   purine ribonucleoside triphosphate binding
GO:0005524      1.000   ATP binding
GO:0016773      1.000   phosphotransferase activity, alcohol group as acceptor
GO:0032559      1.000   adenyl ribonucleotide binding
GO:0032555      1.000   purine ribonucleotide binding
GO:0017076      1.000   purine nucleotide binding
GO:0032553      1.000   ribonucleotide binding
GO:0016301      1.000   kinase activity
GO:0016772      1.000   transferase activity, transferring phosphorus-containing groups
GO:0000166      1.000   nucleotide binding
GO:1901265      1.000   nucleoside phosphate binding
GO:0036094      1.000   small molecule binding
GO:0097367      1.000   carbohydrate derivative binding
GO:0030554      1.000   adenyl nucleotide binding
GO:0005488      1.000   binding
GO:0003824      1.000   catalytic activity
GO:1901363      1.000   heterocyclic compound binding
GO:0097159      1.000   organic cyclic compound binding
GO:0043167      1.000   ion binding
GO:0016740      1.000   transferase activity
--------------------------------------------------
GO:0004674      0.373   protein serine/threonine kinase activity
GO:0016787      0.012   hydrolase activity
GO:0004871      0.002   signal transducer activity
```

Scores below `0.5` are interpreted as negative predictions. Only non-zero scores are shown. 

Use CTRL-C to leave the interactive mode, for example when running a sensitivity analysis next.


## Sensitivity Analysis

Wait until the terminal opens and enter

```bash
bash /code/sense.sh
```

to start the sensitivity analysis mode. 

**Example:**

A prompt will open where you can enter a protein sequence, e.g the sequence of Src-kinase kinase ([PDB 6F3F](https://www.rcsb.org/structure/6f3f))

```bash
Please enter the sequence to analyze:

GARMVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMNKGSLLDFLKGETGKYLRLPQLVDMSAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL
```

and the GO-terms of interest in comma-seperated form, e.g.

```bash
Please enter the GO terms to analyze sperarated by commas:

GO:0016301,GO:0005524
```

The sensitivity analysis can take a few minutes. Results will be written to a tab-separated .txt file and a plot in .png format. To view and download these, click on the 'Jupyter' logo and move to the directory 'results/'. 

There you will find the sensitivities.txt, e.g.

```txt
Pos	AA	sec	dis	GO:0016301	GO:0005524
-1	wt	wt	wt	0.0	0.0
0	G	.	_	0.6929230000000004	0.1063629999999982
1	A	.	_	0.01167100000000332	-0.12102099999999892
2	R	.	_	1.0941490000000016	-0.7712800000000009
3	M	.	_	0.9523300000000036	0.6096879999999985
4	V	.	_	1.514687000000002	1.8048670000000016
5	T	.	_	-0.01615099999999714	0.1625770000000024
6	T	.	_	-0.09454199999999703	0.202874999999998
...
```

**Note:** The output has more than 300 lines and was truncated here.

Each Line contains information about one position in the protein sequence.
Its index is given under **Pos** and the respective amino acid under **AA**.
For each GO term **GO:_______**, there is one column containing the sensitivity value
in that position for that GO term.
The second line and the third and fourth columns ("sec", "dis") are for technical reasons and can be ignored.

**Note** Sequences far off the protein sequence space as covered by SwissProt might give unexpected results. The same applies to very short sequences since DeeProtein was trained on sequences with more than 150 amino acids.


Additionally, a plot of the sensitivity values along the sequence is produces and saves as sensitivities.png, e.g.

<img src="examples/sensitivity_1D.png">

N-terminus enlarged:
<img src="examples/sensitivity_1D_short.png">

You can download these by marking them and clicking 'Download'.


# Visualization in PyMOL

For visualization of the sensitivity data, we provide a plugin for PyMOL in [/DeeProtein/pymol/visualize_sensitivity.py](/DeeProtein/pymol/visualize_sensitivity.py).
You need PyMOL installed on your computer and our plugin to perform the visualization as follows. PyMOL must have access to the following packages:
* [PyMOL 1.7+](https://pymol.org/2/)
* [numpy 1.14.3](http://www.scipy.org/install.html)
* [pandas 0.23.0](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [Bio 1.70](https://biopython.org/)

The plugin assumes the sensitivity values to be in a 'labels' directory, therefore  move sensitivities.txt to a directory called 'labels'

```bash
mkdir labels
mv sensitivities.txt labels/
```

Next, import the plugin in PyMOL

```python
PyMOL> run visualize_sensitivity.py
```

and color the structure according to one of the GO-terms sensitivites (e.g. kinase activity):

```python
PyMOL> color_sensitivity(file="sensitivities.txt", column="GO:0016301", on_pdb="6F3F", on_chain="A", show_hetatm=True, normalize=False, min_val=-5, max_val=5)
```

position the protein and save an image

```python
PyMOL> png sensitivity_3D.png
```

<img src="examples/sensitivity_3D.png" height="350">

For details see the section [Visualization in PyMOL](#visualization-in-pymol).


# Setup

This and the following sections describe how to setup and deploy DeeProtein.

Download the GeneOntology
```bash
$ wget http://purl.obolibrary.org/obo/go.obo
```

Clone this git repository:
   ```bash
   $ git clone https://github.com/juzb/DeeProtein && cd DeeProtein
   ```


## Requirements

- [Python 3.5+](https://www.python.org/downloads/release/python-350/)
- [Cuda 8.0+](https://developer.nvidia.com/cuda-80-ga2-download-archive)
- [Tensorflow 1.7](https://www.tensorflow.org/api_docs/)
- [Tensorlayer 1.11.0](http://tensorlayer.readthedocs.io/en/latest/)
- [GO-Tools](https://github.com/tanghaibao/goatools)
- [pandas 0.23+](https://pandas.pydata.org/pandas-docs/stable/install.html)
- [BioPython 1.7.4](https://biopython.org/)
- [SciPy 0.19+](http://www.scipy.org/install.html)
- [scikit-learn 0.20](http://scikit-learn.org/stable/install.html#)

### Optional
- [PyMOL 1.7+](https://pymol.org/2/)
- [jupyter 1.0+](https://jupyter.org/install)

## General Information

**Sensitivity data**
Sensitivity data is stored in our [zenodo repository](https://doi.org/10.5281/zenodo.1402816).

**DeeProtein Weights**
Weights for DeeProtein are stored in our [zenodo repository](https://doi.org/10.5281/zenodo.1402827).


# Usage
**Note:**

The following usage guide has been tailored to meet the requirements of the code-ocean compute capsule.
It allows the execution of DeeProtein's functionality:
- Inference and classification of a given protein sequence
- Occlusion based sensitivity analysis of a given protein sequence
- Visualization of the computed sensitivity traces

In order to work with DeeProtein in greater detail, i.e. to alter and/or combine models please refer to the `main.sh` script in our (GitHub repository)[https://github.com/juzb/DeeProtein]


## Inference with DeeProtein

Download the weights for DeeProtein from our [zenodo repository](https://doi.org/10.5281/zenodo.1402827)
```bash
$ wget https://zenodo.org/record/2574979/files/complete_model_0.npz
mv complete_model_0.npz complete_model.npz
```

1. Specify the options in config.JSON, see [FLAGS explanations](#flags).
   ```bash
   $ vim config.JSON
   ```
   Alternatively, options can be superseded by specifying the appropriate FLAG in the function call (as seen below).

2. To infer a sequence on the pretrained DeeProtein Model use the sense.sh script as explained in the [Classification](#classification) section.


## Sensitivity Analysis

Use the sense.sh file as described in the [Sensitivity Analysis](#sensitivity-analysis) section or

1. Write a masked_dataset.txt file having the following format:
   Each line consists of
   ```txt
    <PDB ID>;<Chain ID>;<GO:1>,<GO:2>, ...;<Sequence>;<Secondary structure>;<dis string>
   ```

   e.g.

   ```txt
    6F3F;A;GO:0016301,GO:0005524;QRKLEALIRDPRSPINVESLLDGLNSLVLDLDFPALRKNKNIDNFLNRYEKIVKKIRGLQMKAEDYDVVKVIGRGAFGEVQLVRHKASQKVYAMKLLSKFEMIKRSDSAFFWEERDIMAFANSPWVVQLFCAFQDDKYLYMVMEYMPGGDLVNLMSNYDVPEKWAKFYTAEVVLALDAIHSMGLIHRDVKPDNMLLDKHGHLKLADFGTCMKMDETGMVHCDTAVGTPDYISPEVLKSQGGDGYYGRECDWWSVGVFLFEMLVGDTPFYADSLVGTYSKIMDHKNSLCFPEDAEISKHAKNLICAFLTDREVRLGRNGVEEIKQHPFFKNDQWNWDNIRETAAPVVPELSSDIDSSNFDDITFPIPKAFVGNQLPFIGFTYYR;...............................................................................................................................................................................................................................................................................................................................................................................................;_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
   ```

   where, sequence, secondary structure and dis string are of the same length.
   If the secondary structure is unknown a string containing only points is valid.
   If the dis string is unknown, a string containing only underscores is valid.


2. Score the occluded sequences using

   ```bash
   $ python DeeProtein.py -mask=True -restore=True -restorepath=path/to/saves -model_to_use=model/model.py -v path/to/test/parent-directory-of-masked_dataset
   ```
   then
   ```bash
   $ python DeeProtein/scripts/calculate_sensitivity.py  path/to/test/parent-directory-of-masked_dataset/masked_dataset.txt path/to/test/parent-directory-of-masked_dataset/masked_dataset.txt
   ```
   
3. Visualize the computed sensitivities in a sensitivity trace plot:

   ```bash
   $ python DeeProtein/scripts/plot_sensitivity.py path/to/test/parent-directory-of-masked_dataset/masked_dataset.txt "GO-to-plot" /outputdir/sensitivity_1D.png 
   ```

# FLAGS

A number of FLAGS is available in `config.JSON` to specify the behavior of DeeProtein, both for inference and training.
While all FLAGS may be superseded in the function calls, we recommend to set up the config.JSON prior to usage:

FOR INFERENCE AND SENSITIVITY ANALYSIS:
* `info_path`        `str, the path where logs and the metrics calculation related files should be saved
* `infer`        'str', either `True` or `False` (as 'str'), whether to run the model in inference mode.
* `mask`        'str', either `True` or `False` (as 'str'), if `True` enables the model to perform sensitivity analysis on a masked dataset specified in `validdata`
* `restorepath`        'str', path to directory which contains a complete_model.npz file, the saves from which to restore the model
* `print_all`        'str', either `True` or `False` (as 'str'), whether to print ALL (1000) Go term scores in interactive inference. if `False` only the top 20 are printed.
* `restore`        'str', either `True` or `False` (as 'str'), whether to restore the model from `restorepath` or train a new model. Defaults to `True`
* `godagfile`        'str', the path to the Gene-Ontology .obo file
* `gofile`        'str', the path to file specifying the GO-terms the model should consider. One GO per line. Available in examples/go_file_1000.txt

FOR TRAINING A NEW MODEL:
* `validdata`        'str', the path to the datasets as .csv. Alternative use in 'mask' mode (i.e. sensitivity analysis): path to masked dataset file
* `traindata`        'str', the path to the datasets as .csv
* `test`        'str', either `True` or `False` (as 'str'), whether to activate test mode. Test data to be specified under `validdata`
* `compare`        'str', either `True` or `False` (as 'str'), Compares multiple models based on their dumps, paths specified comma-separated in `validdata`
* `model_to_use`        'str', the path to the .py holding the model to use in DeeProtein. Default: `model/model.py`
* `logging_level`        'str', One of \[INFO, DEBUG, WARNING\] the basic level on which the logger should operate.
* `modelname`        'str', the name of the Model. Default 'DeeProtein'
* `nblocks`        'int', number of residual blocks to build. Defaults to 30, given weights do only comply with that number.
* `print_num_params`        'str', either `True` or `False` (as 'str'). Printing the number of parameters is time intensive but interesting. Choose wisely...
* `windowlength`        'int', Max. Number of sequence position to consider. Longer sequences are clipped. Defaults to `1000`.
* `learningrate`        'float', initial learning rate for the training of DeeProtein
* `epsilon`        'float', epsilon parameter for the ADAM optimizer
* `batchsize`        'int', batchsize. We recommend 64 for training. Defaults to `1` (for inference).
* `nepochs`        'int', number of epochs to train the model. Defaults to `20`.
* `valid_after`        'int', number of batches after which to perform an early run over the valid set.
* `early_valid_size`        'int', number of batches to use in early validation
* `keep_plots`        'str', either `True` or `False` (as 'str'), whether to keep or overwrite plots from each validation run


# Visualization in PyMOL

   ```python
  PyMOL> color_sensitivity(file,
                            column=None,
                            show_hetatm=True,
                            show_chains=True, 
                            on_chain=None,    
                            on_pdb=None,      
                            reload=True,     
                            normalize=True,   
                            min_val=-1,      
                            max_val=1        
                           )      
   ```

   Use to automatically color a structure by sensitivity. Is called from the PyMOL console.
   If the 3D structure needed to show a given proteins sensitivity is not
   available in the specified 'prot_path', the PDB file is automatically downloaded.

   * `file`            sensitivity value file, name should be '\<prefix\>_\<PDB ID\>_\<Chain ID\>_\<suffix\>.\<ending, usually txt\>
   * `column`          column to color according to
   * `show_ligands`    whether to show hetatm-flagged atoms by default, e.g. ligands
   * `show_chains`     whether to show other chains, e.g. interacting proteins in the pdb file
   * `on_chain`        redirect coloring to other chain the file-name suggests
   * `on_pdb`          redirect coloring to other pdb file
   * `reload`          whether to reload the 3D structure, can take a few seconds
   * `normalize`       whether to normalize the values to color, keeps zero at zero, does not work well, if there are few outliers
   * `min_val`         instead of normalization, the upper and the lower border for coloring can be given here \n
   * `max_val`         min_val and max_val must have the same absolute value, if a value of 0 shall be colored white
