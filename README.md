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


