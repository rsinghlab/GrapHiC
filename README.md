# GrapHiC

## Abstract 
Hi-C experiments allow researchers to study and understand the 3D genome organization and its regulatory function. Unfortunately, sequencing costs and technical constraints severely restrict access to high-quality Hi-C data for many cell types. Existing deep learning methods use the convolutional neural network (CNN) based frameworks to improve the resolution of Hi-C contact maps with low read coverage. However, these methods treat Hi-C maps as images, which imposes a strict 2D euclidean structure that, in reality, represents a 3D structure, resulting in low performance on real-world sparse Hi-C datasets. We propose GrapHiC, which utilizes a Graph Auto-Encoder network to generate high-quality Hi-C contact maps. We formulate the Hi-C data as a graph, where each node represents a binned genomic region, and we assign edge weights using the observed Hi-C contact reads between two regions. Our formulation captures the 3D structure accurately and allows us to integrate easy-to-obtain 1D genomic signals as node features. This auxiliary data provides cell-type-specific information to impute missing reads when the input Hi-C data is very sparse. Our experiments on datasets with varying sparsity levels show that GrapHiC generates better quality Hi-C maps than state-of-the-art methods, especially for very sparse input Hi-C data. More importantly, our framework imputes the Hi-C map reliably in a cell line with missing input Hi-C data by using its genomic signals and a Hi-C map from another cell line. Thus, our generalizable framework can make the analysis of high-quality Hi-C data more accessible for many cell types.


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

### Setting up CUDA, PYTORCH and PYTORCH GEOMETRIC
At the time of writing this readme, the stable versions of CUDA, Pytorch and PyGeometric werent compatible with each other. So I managed to figure out a way to make them run with each other by downgrading to older stable versions of all three of them. If you already have a setup that runs fine ignore this step and skip to the step of installing rest of the packages. 

I managed to get everything running on CUDA version 11.3 (I needed 11.3 because my university cluster was predominantly Ampere architecture GPUs). 

I fortunately did not have to install CUDA myself they came nicely packaged on my cluster, but installation these days is very straightforward. I would follow instruction on this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux). After years of struggling with Nvidia drivers, I recommend to only install the stable release drivers, the latest (or nightly releases) are very buggy and generally are typically are unstable for Linux machines. 


To install Pytorch, I ran this pip installation command,
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
This should install pytorch 1.11.0 and its sub-modules compatible with CUDA version 11.3. I followed the installation instructions on pytorch's official install instruction [page](https://pytorch.org/get-started/previous-versions/). This page contains links on how to install specific version of pytorch versions. 

Finally, to install pytorch geometric I ran this command, 
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

This should install appropriate required packages of pygeometric and ideally after this step everything should run fine without broken symbolic links. 

### Installing rest of the packages
For rest of the packages just run command and the pip should take care of the dependencies and install rest of the packages. Note, I have included the pytorch and pygeometric packages in this requirements file and maybe on a later date this is all you need to run to install the required packages. 

```
pip install -r requirements.txt
```

After this everything else should work fine. 


## Updating static paths in the codebase
I used static file paths defined in the src/utils.py for almost the entire testing and development phase of this project. There are two main paths in this file that need to be updated. 
1) BASE_DIRECTORY path points or stores the absolute path of the directory where you have pulled the GrapHiC source code. Update this path with the ouput of the 'pwd' command. 
2) DATA_DIRECTORY path points to the folder that stores all the downloaded, parsed and generated data. Ideally I would point this path to a device that has ample storage capacity (~150GBs). 

I have many wrapper functions that ensure that all the files and folders are in the correct order. However, if you still have issues setting your paths please feel free to reach out. 


## Datasets
All the remote paths and the local paths of all the datasets used in this project are defined in the src/utils.py file. We download 3 high-resolution Hi-C datasets, 7 low-resolution Hi-C datasets and 42 epigenetic datasets through various portals. If you want to add datasets for other cell lines, add a new entry in the datasets path dictionary similar to the existing entries. For Hi-C datasets you only need to define the label of the dataset that is in format $cell-line-source-id$ and link to the remote database that contains that dataset. For epigenetic datasets you need to put it under a sub-dictionary of the $cell-line$ labeled with appropriate experiment id such as H3K27ME3. 


## Training GrapHiC 
Once you have installed all the GrapHiC requirements and setup the paths accordingly, you can run the command:
```
python training_scripts/graphic_training.py
```
This command downloads all the necessary datasets both Hi-C and Auxiliary signals, pre-processes them and converts them into a dataset that is fed into the training pipeline. This trains the GrapHiC model with the parameters specified in the 'parameters.py' file and stores the weights in the specified weights directory. This function also evaluates GrapHiC on all five GM12878 cell lines on test chromosomes. The training and testing scripts for HiCReg, HiCNN and different versions of GrapHiC follow the same workflow and we have created distinct .py files in the same folder. 

## Imputing Hi-C reads with GrapHiC


