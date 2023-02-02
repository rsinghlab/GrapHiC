# GrapHiC

## Abstract 
Hi-C experiments allow researchers to study and understand the 3D genome organization and its regulatory function. Unfortunately, sequencing costs and technical constraints severely restrict access to high-quality Hi-C data for many cell types. Existing deep learning methods use the convolutional neural network (CNN) based frameworks to improve the resolution of Hi-C contact maps with low read coverage. However, these methods treat Hi-C maps as images, which imposes a strict 2D euclidean structure that, in reality, represents a 3D structure, resulting in low performance on real-world sparse Hi-C datasets. We propose GrapHiC, which utilizes a Graph Auto-Encoder network to generate high-quality Hi-C contact maps. We formulate the Hi-C data as a graph, where each node represents a binned genomic region, and we assign edge weights using the observed Hi-C contact reads between two regions. Our formulation captures the 3D structure accurately and allows us to integrate easy-to-obtain 1D genomic signals as node features. This auxiliary data provides cell-type-specific information to impute missing reads when the input Hi-C data is very sparse. Our experiments on datasets with varying sparsity levels show that GrapHiC generates better quality Hi-C maps than state-of-the-art methods, especially for very sparse input Hi-C data. More importantly, our framework imputes the Hi-C map reliably in a cell line with missing input Hi-C data by using its genomic signals and a Hi-C map from another cell line. Thus, our generalizable framework can make the analysis of high-quality Hi-C data more accessible for many cell types.




## How to setup the enviroment
I highly recommend that you use make an anaconda environment (or a virtual environment) before you begin setting up the code base. I have given the install instruction for both, follow whatever is easier in your case. 

### Installing using Anaconda
Make sure you are using CUDA version 10.2 for this installation process. I would follow instruction on this [link](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux). I would highly recommend that you make sure you have the correct version of Nvidia drivers installed as well, otherwise this can quickly become very complicated to debug. 


After you have pulled this repository, 

```
conda env create --name grapHiC --file=environments.yml
```

I have commited my anaconda enviroment.yml file that should in theory install appropriate packages and ensure you have no dependency conflicts. I have made that .yml file for an even broader audience that has glibc of version 2.17 and for whatever reasons they are unable to update it (as it was in my cluster's case). 

Finally, activate the environment and that should be it. 
```
conda activate grapHiC
```


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
## Setting up CUDA, PYTORCH and PYTORCH GEOMETRIC
At the time of writing this readme, the stable versions of CUDA, Pytorch and PyGeometric werent compatible with each other. So I managed to figure out a way to make them run with each other by downgrading to older stable versions of all three of them. If you already have a setup that runs fine ignore this step and skip to the step of installing rest of the packages. 

I managed to get everything running on CUDA version 11.3 (I needed 11.3 because my university cluster was predominantly Ampere architecture GPUs). 

I fortunately did not have to install CUDA myself they came nicely packaged on my cluster, but installation these days is very straightforward. I would follow instruction on this [link](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux). After years of struggling with Nvidia drivers, I recommend that you always and only install the stable release drivers, the latest (or nightly releases) are very buggy and generally are very unstable for Linux machines. 


To install Pytorch, I ran this pip installation command,
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
This should install pytorch 1.11.0 and its sub-modules compatible with CUDA version 11.3. I followed the installation instructions on pytorch's official install instruction [page](https://pytorch.org/get-started/previous-versions/). This page contains links on how to install specific version of pytorch versions. 

Finally, to install pytorch geometric I ran this command, 
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
Looking in links: https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

This should install appropriate required packages of pygeometric and ideally after this step everything should run fine without broken symbolic links. 

### Installing rest of the packages
For rest of the packages just run command, 

```
pip install -r requirements.txt
```
After this everything else should work fine. 


## Setting up paths inside the codebase
All the static paths used during the testing and development phase are stored in the file GrapHiC/src/utils.py. There are x main paths that need to be updated. I made wrapper scripts that do folder creation and folder deletions and they expect full/absolute paths so whatever you decide to update your paths with make sure they are absolute paths. Or modify the implementation of those functions whatever is easier for your use case. 
1) BASE_DIRECTORY path points or stores the absolute path of the directory where you pulled this codebase. If you have changed the path of your terminal since the start of this installation process running 'pwd' command would give you the absolute path. Update the value of BASE_DIRECTORY to the output of pwd command. 
2) HIC_FILES_DIRECTORY, PARSED_HIC_FILES_DIRECTORY and DATASET_DIRECTORY can be anywhere in your filesystem just make sure that whatever filesystem they are on, it has enough space to store ~200GBs of both raw and processed .hic datasets. 
3) WEIGHTS_DIRECTORY and GENERATED_DATA_DIRECTORY are currently setup to be automatically adjusted based on the BASE_DIRECTORY but if you want to store these somewhere else feel free to change them as well. WEIGHTS_DIRECTORY stores the weights of the models (for each epoch in current implementation) and GENERATED_DATA_DIRECTORY all the generated visualizations.  


### Training and Testing GrapHiC
Once you have installed all the GrapHiC requirements and setup the paths accordingly, you can run the command:
```
python grapHiC.py
```

This command downloads all the necessary datasets both Hi-C and Auxiliary signals, pre-processes them and converts them into a dataset that is fed into the training pipeline. This trains the GrapHiC model with the parameters specified in the 'parameters.py' file and stores the weights in the specified weights directory. This function also evaluates GrapHiC on all five GM12878 cell lines on test chromosomes. The training and testing scripts for HiCReg, HiCNN and different versions of GrapHiC follow the same workflow and we have created distinct .py files for each having a similar workflows. 

