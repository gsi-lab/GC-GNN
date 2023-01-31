# Combining Group-Contribution concept and Graph Neural Networks Towards Interpretable Molecular Property Models
***


## Background
***
This code is the basis of our work submitted to *Journal of Chemical Information and Modeling*, aiming to 
integrate *a-priori* knowledge, i.e. group contribution theory, with graph neural networks and attention mechanism, and bring more insights in the prediction of thermodynamic properties. This is an **alpha version** of the models used to generate the resuls published in:

[Combining Group-Contribution Concept and Graph Neural Networks Toward Interpretable Molecular Property Models](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01091)

Adem R. N. Aouichaoui, Fan Fan, Seyed Soheil Mansouri, Jens Abildskov, and Gürkan Sin
Journal of Chemical Information and Modeling Article ASAP
DOI: 10.1021/acs.jcim.2c01091


## Prerequisites and dependencies
***
The code requires the following packages and modules:\
\
bayesian_optimization==1.2.0\
dgl==0.9.0\
dgl_cuda11.3==0.8.1\
matplotlib==3.5.1\
numpy==1.23.2\
openTSNE==0.6.2\
pandas==1.4.3\
prettytable==2.1.0\
rdkit==2022.3.5\
scikit_learn==1.1.2\
seaborn==0.11.1\
torch==1.8.1\
tqdm==4.64.0\
umap==0.1.1\
umap_learn==0.5.3


## Usage
***
### Fragmentation and Cache Files
The relevant scheme for fragmentation and their corresponding SMARTs are kept in './datasets/MG_plus_reference.csv'.

Meanwhile the folders storing cache files and error logs would be created automatically in the root directory after the first time of scripts running.
### Training and SplittingSeed Tuning
A command line for training a non-frag based model and tuning the splitting seed:
```commandline
$ python Seed_Tuning.py 
```
A command line for training a frag based model and tuning the splitting seed:
```commandline
$ python Seed_Tuning_Frag.py 
```
The results of every attempt containing metrics on three folds are automatically printed in './output/'.
### Optimization
To carry out a single-objective bayesian optimization on a non-frag-based model, do:
```commandline
$ python new_optimization_nonfrag.py
```
To carry out a single-objective bayesian optimization on a frag-based model, do:
```commandline
$ python new_optimization_frag.py
```
### Generate Ensemble Models
To generate ensemble models with random initializations on non-frag-based models, do:
```commandline
$ python ensembling_nonfrag.py
```
To generate ensemble models with random initializations on frag-based models, do:
```commandline
$ python ensembling_frag.py
```
Note that the default ensemble size is 100. And the trained models would be saved in folders named by the model names under './library/Ensembles/', which would be created when these scripts operating.

After the ensemble models generated, these two scripts need to be carried out to print the predictions of every compounds in dataset together with their latent representations.
```commandline
$ python ensemble_frag_compound.py
$ python ensemble_nonfrag_compound.py
```
All these outputs are available under the folders where models are saved in.
## Contribution
***
Adem Rosenkvist Nielsen Aouichaoui ([arnaou@kt.dtu.dk](arnaou@kt.dtu.dk))

Fan Fan ([chn.tz.fanfan@outlook.com](chn.tz.fanfan@outlook.com))

## Licence
***
Check the licence file

