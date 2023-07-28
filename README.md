# GrapHiC

## Abstract 
Hi-C experiments allow researchers to study and understand the 3D genome organization and its regulatory function. Unfortunately, sequencing costs and technical constraints severely restrict access to high-quality Hi-C data for many cell types. Existing frameworks rely on a sparse Hi-C dataset or cheaper-to-acquire ChIP-seq data to predict Hi-C contact maps with high read coverage. However, these methods fail to generalize to sparse or cross-cell-type inputs because they do not account for the contributions of epigenomic features or the impact of the structural neighborhood in predicting Hi-C reads. We propose GrapHiC, which combines Hi-C and ChIP-seq in a graph representation, allowing more accurate embedding of structural and epigenomic features. Each node represents a binned genomic region, and we assign edge weights using the observed Hi-C reads. Additionally, we embed ChIP-seq and relative positional information as node attributes, allowing our representation to capture structural neighborhoods and the contributions of proteins and their modifications for predicting Hi-C reads. Our evaluations show that GrapHiC generalizes better than the current state-of-the-art on cross-cell-type settings and sparse Hi-C inputs. Moreover, we can utilize our framework to impute Hi-C reads even when no Hi-C contact map is available, thus making high-quality Hi-C data more accessible for many cell types.

![alt text](https://github.com/rsinghlab/GrapHiC/blob/main/arch.jpg?raw=true)


## How to setup the enviroment
I highly recommend that you use make a virtual environment before you begin setting up the code base. 


### Creating Virtual Environment
After you have pulled this repository, 

```
cd GrapHiC
```

Then create a virtual environment in the directory.

```
python3 -m venv grapHiC
```

Once you have setup the virtual environment activate it so we can finally start our package installation process. 

```
source grapHiC/bin/activate
```

### A note on CUDA and GCC
We are using CUDA version 11.7.1 and GCC version 10.2; if there are installation bugs, manually resolve the dependencies by installing pytorch and pygeometric corresponding to the CUDA and GCC you have available. 


### Installing rest of the packages
For rest of the packages just run command and the pip should take care of the dependencies and install rest of the packages. Note, I have included the pytorch and pygeometric packages in this requirements file.

```
pip install -r requirements.txt
```

## Updating static paths in the codebase
I have only two paths that are statically defined in the code base and they refer to the current working directory (directory where you put GrapHiC source code) and the data directory where you wish to store the outputs and datasets. 
Static paths are defined in the parameters.py file and I would update them as follows:
1) BASE_DIRECTORY path should point to the output of running the pwd command.
2) DATA_DIRECTORY path should point to a folder (anywhere) that has ample storage (~150 GBs) capacity available. 

I have many wrapper functions that ensure that all the files and folders are in the correct order. However, if you still have issues setting your paths please feel free to reach out. 

## Datasets
All the remote paths and the local paths of all the datasets used in this project are defined in the src/utils.py file. We download 3 high-resolution Hi-C datasets, 7 low-resolution Hi-C datasets and 42 epigenetic datasets through various portals. If you want to add datasets for other cell lines, add a new entry in the datasets path dictionary similar to the existing entries. For Hi-C datasets you only need to define the label of the dataset that is in format $cell-line-source-id$ and link to the remote database that contains that dataset. For epigenetic datasets you need to put it under a sub-dictionary of the $cell-line$ labeled with appropriate experiment id such as H3K27ME3.


## Training GrapHiC 
Once you have installed all the GrapHiC requirements and setup the paths accordingly, you can run the command:
```
python train_GrapHiC.py
```
This command downloads all the necessary datasets both Hi-C and Auxiliary signals, pre-processes them and converts them into a dataset that is fed into the training pipeline. This trains the GrapHiC model with the parameters specified in the 'parameters.py' file and stores the weights in the specified weights directory. This function also evaluates GrapHiC on all five GM12878 cell lines on test chromosomes. The training and testing scripts for HiCReg, HiCNN and different versions of GrapHiC follow the same workflow and we have created distinct .py files in the same folder. 

## Imputing Hi-C reads with GrapHiC
Once you have retrained GrapHiC (or downloaded the provided weights), you can run the command:
```
python impute_chroms_GrapHiC.py
```
To generate imputed samples for all the datasets and analysis we performed in our manuscript. Similar scripts are provided for HiCNN and HiCReg to impute Hi-C data for all the datasets. 

## Running analysis
Once you have generated all the data, you can run the command to run analysis scripts and evaluate and compare the performance of GrapHiC against other methods:
```
python evaluation/evaluate.py
```
This script would generate aggregate results for all the test chromosomes in the RESULTS_DIRECTORY folder defined in the utils.py file along with visualization of the generated chromosomes. Unfortunatly, for Hi-C similarity metric we generate script files that run 3DChromatin_ReplicateQC replicate pipeline. 


## Online App
GrapHiC is also available as an app on [superbio.ai](https://app.superbio.ai/apps/196?id=63888da92ffa50c6deecdca2).







