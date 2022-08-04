# GrapHiC
A graph neural network based approach to upscale quality of HiC matrices.


Make sure you update all the hard coded paths in the src/utils.py to where ever you feel is appropriate. Secondly, the requirements file contain an expansive set of packages so I would recommend that you make a virtual environment.



## How to setup the enviroment
I highly recommend that you make a virtual enviroment or (anaconda enviroment) before you begin setting up the code base. 


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




