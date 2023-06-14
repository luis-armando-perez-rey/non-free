# Equivariant Isomorphic Networks (EquIN)

This repository contains the code used to run the experiments for the paper Equivariant Representation Learning in the Presence of Stabilizers. 

## Installation

The code is written in Python 3.6. To install the required packages, run

``` pip install -r requirements.txt ```

## Datasets

To organize the datasets into pairs for training run the code:
```angular2html
bash generate_dataset.sh DATASET_NAME
```
where DATASET_NAME is the name of the dataset. The datasets used in the paper are: ```arrows```, ```arrows_colors```, ```double_arrows```, ```symmetric_solids```, ```modelnet```.

## Running the experiments

To run the experiments use the following command: 
``` 
bash run.sh CONFIG_FILENAME [SEED] 
```
where CONFIG_FILENAME is the name of the config file in the configs folder and SEED is the seed used for the experiment. If no seed is provided, the default seed is 17. The seeds used in the experiments of the paper are 17, 19, 28, 42, 58.
The experiment configs are stored in the configs folder and are as follows: 
 


| Config File       | Model | Dataset       | Summary                                                                                                                                         |
|-------------------|-------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| single_arrows     | EquIN | Arrows        | Experiment that is trained with a single object made of 1,2,3,4 or 5 arrows. The EquIN is trained across multiple N values 1,2,3,4,5,6,7,8,9,10 |
| all_arrows        | EquIN | Arrows        | Experiment trained with 1,2,3,4,5 arrow objects at the same time. The value $N$ is iterated across the values 1, 5, 10                          |    
| all_arrows_colors | EquIN | Arrows Colors | Experiment trained with 1,2,3,4,5 arrow objects with different colors. The value $N$ is iterated across the values 1, 5, 10                     |
| chamfer_reg       | EquIN | Arrows        | Experiment that uses 1,2,3,4,5 arrow objects and iterates across different values of $\lambda={0.001, 0.01, 0.1, 1.0, 5.0, 10}                  |
| ENR_arrows        | ENR   | Arrows        | Experiment that uses the same dataset as all_arrows but trained with ENR                                                                        |
| ENR_arrows_colors | ENR   | Arrows Colors | Experiment that uses the same dataset as all_arrows_colors but trained with ENR                                                                 |
| all_torus         | EquIN | Double Arrows | Experiment trained with the double arrows dataset using $N$ with 1,6,15,20                                                                      |
| all_solids        | EquIN | Solids        | Experiment trained with the solids dataset using $N$ with 1,12,24,60,80                                                                         |
| ENR_solids        | ENR   | Solids        | Experiment that uses the same dataset as all_solids but trained with ENR                                                                        |
| all_modelnet      | EquIN | ModelNet      | Experiment trained with the ModelNet dataset using $N$ with 1, 4, 10                                                                            |
| ENR_modelnet      | ENR   |       ModelNet     | Experiment that uses the same dataset as all_modelnet but trained with ENR                                                                      |
