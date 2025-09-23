# Physics-Based Animation: The Finite Element Method
In this assignment you will learn to implement the finite element method for both volumetric and thin shell objects.

**WARNING:** Do not create public repos or forks of this assignment or your solution. Do not post code to your answers online or in the class discussion board. Doing so will result in a 20% deduction from your final grade. 

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

## Tools You Will Use
1. [NVIDIA Warp](https://github.com/NVIDIA/warp) -- python library for kernel programming
2. [PyTorch](https://pytorch.org/) -- python library for array management, deep learning etc ...
3. [SymPy](https://www.sympy.org/en/index.html) -- python symbolic math package
   
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
## Assignment Structure and Instructions
1. You are responsible for implementing all functions found in the [assignments](./assignment) subdirectory.
2. The [tests](./tests) subdirectory contains the scenes, specified as python files,  we will validate your code against.
3. The test_output subdirectory contains output from the solution code that you can use to validate your code. This output comes in two forms. (1) **USD (Universal Scene Description)** files which contain simulated results. These can be played back in any USD viewer. I use [Blender](https://www.blender.org/). You can output your own simulations as USD files, load both files in blender and examine the simulations side-by-side. (2) Two.pt files which contains the global mass matrix (as a dense matrix) for the one_tet_fall.py scene and the bunny_fall.py scene which you can [load](https://docs.pytorch.org/docs/stable/generated/torch.load.html) and compare your own code to.

In this assignment you will get a chance to implement one of the  gold-standard methods for simulating elastic objects -- the finite element method (FEM). Unlike the particles in the previous [assignment](https://github.com/dilevin/CSC417-a2-mass-spring-3d), the finite-element method allows us compute the motion of continuous volumes of material. This is enabled by assuming that the motion of a small region can be well approximated by a simple function space. Using this assumption we will see how to generate and solve the equations of motion.   

FEM has wonderfully practical origins, it was created by engineers to study [complex aerodynamical and elastic problems](https://en.wikipedia.org/wiki/Finite_element_method) in the 1940s. My MSc supervisor used to regale me with stories of solving finite element equations by hand on a chalkboard. With the advent of modern computers, its use as skyrocketed.  

FEM has two main advantages over mass-spring systems. First, the behaviour of the simulated object is less dependent on the topology of the simulation mesh. Second, unlike the single stiffness parameter afforded by mass spring systems, FEM allows us to use a richer class of material models that better represent real-world materials. 

## Resources

Part I of this [SIGGRAPH Course](http://www.femdefo.org), by Eftychios Sifakis and Jernej Barbic, is an excellent source of additional information and insight, beyond what you will find below. 


## Admissable Code and Libraries
You are allowed to use SymPy for computing formulas for integrals, derivatives and gradients. You are allowed to use any functions in the warp and warp.spare packages. You ARE NOT allowed to use code from other warp packages like warp.fem. You are not allowed to use any of warps specialized spatial data structures for storing meshes, volumes or doing spatial subdivision. You cannot use code from any other external simulation library.  

## Hand-In
We will collect and grade the assignment using MarkUs, link **coming soon**

## Late Penalty
The late penalty is the same as for the course, specified on the main github page. 
