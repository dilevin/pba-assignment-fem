# Physics-Based Animation: The Finite Element Method
In this assignment you will learn to implement the finite element method for both volumetric and thin shell objects.

**WARNING:** Do not create public repos or forks of this assignment or your solution. 

## Checking out the code and setting up the python environment
These instructions use [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) for virtual environment. If you do not have it installed, follow the 
instructions at the preceeding link for your operating system

Checkout the code ```git clone git@github.com:dilevin/physics-based-animation-fem.git {ROOT_DIR}```, where **{ROOT_DIR}*** is a directory you specify for the source code. 

Next create a virtual environment and install relevant depencies install python dependencies.
```
cd {ROOT_DIR}
conda create -n csc417  python=3.12 -c conda-forge
conda activate csc417
pip install -e . 
```
Optionally, if you have an NVIDIA GPU you might need to install CUDA if you want to use the GPU settings
```
conda install cuda -c nvidia/label/cuda-12.1.0
```
Assignment code templates are stored in the ```{ROOT_DIR}/assginment``` directory. 

## Running the Assignment Code
```
cd {ROOT_DIR}
python main.py --scene=tests/{SCENE_PYTHON_FILE}.py
```
By default the assignment code runs on the cpu, you can run it using your GPU via:
```
python main.py --scene=tests/{SCENE_PYTHON_FILE}.py --device=cuda
```
Finally, the code runs, headless and can write results to a USD file which can be viewed in [Blender](https://www.blender.org/):
```
python main.py --scene=tests/{SCENE_PYTHON_FILE}.py --usd_output={FULL_PATH_AND_NAME}.usd --num_steps={Number of steps to run}
```
