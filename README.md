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

**WIKNDOWS NOTE:*** If youb want to run the assignmetns using your GPU you may have to force install torch with CUDA support using ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

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
3. The [test_output](./test_output) subdirectory contains output from the solution code that you can use to validate your code. This output comes in two forms. (1) **USD (Universal Scene Description)** files which contain simulated results. These can be played back in any USD viewer. I use [Blender](https://www.blender.org/). You can output your own simulations as USD files, load both files in blender and examine the simulations side-by-side. (2) Two.pt files which contains the global mass matrix (as a dense matrix) for the one_tet_fall.py scene and the bunny_fall.py scene which you can [load](https://docs.pytorch.org/docs/stable/generated/torch.load.html) and compare your own code to.

In this assignment you will get a chance to implement one of the  gold-standard methods for simulating elastic objects -- the finite element method (FEM). Unlike the particles in the previous [assignment](https://github.com/dilevin/CSC417-a2-mass-spring-3d), the finite-element method allows us compute the motion of continuous volumes of material. This is enabled by assuming that the motion of a small region can be well approximated by a simple function space. Using this assumption we will see how to generate and solve the equations of motion.   

FEM has wonderfully practical origins, it was created by engineers to study [complex aerodynamical and elastic problems](https://en.wikipedia.org/wiki/Finite_element_method) in the 1940s. My MSc supervisor used to regale me with stories of solving finite element equations by hand on a chalkboard. With the advent of modern computers, its use as skyrocketed.  

FEM has two main advantages over mass-spring systems. First, the behaviour of the simulated object is less dependent on the topology of the simulation mesh. Second, unlike the single stiffness parameter afforded by mass spring systems, FEM allows us to use a richer class of material models that better represent real-world materials. 

## Resources

Part I of this [SIGGRAPH Course](http://www.femdefo.org), by Eftychios Sifakis and Jernej Barbic, is an excellent source of additional information and insight, beyond what you will find below. 

## Background

In this assignment you will get a chance to implement one of the  gold-standard methods for simulating elastic objects -- the finite element method (FEM). Unlike the particles in the previous [assignment](https://github.com/dilevin/CSC417-a2-mass-spring-3d), the finite-element method allows us compute the motion of continuous volumes of material. This is enabled by assuming that the motion of a small region can be well approximated by a simple function space. Using this assumption we will see how to generate and solve the equations of motion.   

FEM has wonderfully practical origins, it was created by engineers to study [complex aerodynamical and elastic problems](https://en.wikipedia.org/wiki/Finite_element_method) in the 1940s. My MSc supervisor used to regale me with stories of solving finite element equations by hand on a chalkboard. With the advent of modern computers, its use as skyrocketed.  

FEM has two main advantages over mass-spring systems. First, the behaviour of the simulated object is less dependent on the topology of the simulation mesh. Second, unlike the single stiffness parameter afforded by mass spring systems, FEM allows us to use a richer class of material models that better represent real-world materials. 

## Resources

Part I of this [SIGGRAPH Course](http://www.femdefo.org), by Eftychios Sifakis and Jernej Barbic, is an excellent source of additional information and insight, beyond what you will find below. 

![Armadillo simulated via Finite Element Elasticity](images/armadillo.gif)

## The Finite Element method

The idea of the finite element method is to represent quantities inside a volume of space using a set of scalar *basis* or *shape* functions <img src="images/ec7986eb29346b1eae6b36942b0fb975.svg?invert_in_darkmode" align=middle width=40.76954639999999pt height=24.65753399999998pt/> where <img src="images/14b88c29f249c16ee64da346bc37d777.svg?invert_in_darkmode" align=middle width=50.55236834999999pt height=26.76175259999998pt/> is a point inside the space volume. We then represent any quantity inside the volume as a linear combination of these basis functions:

<p align="center"><img src="images/fe6804f6bc753bbf4631ab6c9542bec5.svg?invert_in_darkmode" align=middle width=141.73209435pt height=47.93392394999999pt/></p>

where <img src="images/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/> are weighting coefficients. Designing a finite element method involves making a judicious choice of basis functions such that we can compute the <img src="images/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/>'s efficiently. Spoiler Alert: in the case of elastodynamics, these <img src="images/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/>'s will become our generalized coordinates and will be computed via time integration. 

## Our Geometric Primitive: The Tetrahedron

For this assignment we will use a [tetrahedron](https://en.wikipedia.org/wiki/Tetrahedron) as the basic space volume. The reason we work with tetrahedra is two-fold. First, as you will see very soon, they allow us to easily define a simple function space over the volume. Second, there is available [software](https://github.com/Yixin-Hu/TetWild) to convert arbitrary triangle meshes into tetrahedral meshes.

![A Tetrahedron](https://upload.wikimedia.org/wikipedia/commons/8/83/Tetrahedron.jpg)  

## A Piecewise-Linear Function Space 

Now we are getting down to the nitty-[gritty](https://en.wikipedia.org/wiki/Gritty_(mascot)) -- we are going to define our basis functions. The simplest, useful basis functions we can choose are linear basis functions, so our goal is to define linear functions inside of a tetrahedron. Fortunately such nice basis functions already exist! They are the [barycentric coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system). For a tetrahedron there are four (4) barycentric coordinates, one associated with each vertex. We will choose <img src="images/c83e439282bef5aadf55a91521506c1a.svg?invert_in_darkmode" align=middle width=14.44544309999999pt height=22.831056599999986pt/> to be the <img src="images/3def24cf259215eefdd43e76525fb473.svg?invert_in_darkmode" align=middle width=18.32504519999999pt height=27.91243950000002pt/> barycentric coordinate. 

Aside from being linear, barycentric coordinates have another desireable property, called the *[Kronecker delta](https://en.wikipedia.org/wiki/Kronecker_delta) property* (or fancy Identity matrix as I like to think of it). This is a fancy-pants way of saying that the <img src="images/c36ef0ba4721f49285945f33a25e7a45.svg?invert_in_darkmode" align=middle width=20.92202969999999pt height=26.085962100000025pt/> barycentric coordinate is zero (0) when evaluated at any vertex, <img src="images/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/>, of the tetrahedron, <img src="images/6e355470c45cc13b0430c284142cf243.svg?invert_in_darkmode" align=middle width=35.29127414999999pt height=22.831056599999986pt/>, and one (1) when evaluated at <img src="images/33bf811999b31b361dfb87ee222620f3.svg?invert_in_darkmode" align=middle width=35.29127414999999pt height=21.68300969999999pt/>. What's the practical implication of this? Well it means that if I knew my function <img src="images/60c7f8444e2f683eb807f4085729add6.svg?invert_in_darkmode" align=middle width=34.73748959999999pt height=24.65753399999998pt/>, then the best values for my <img src="images/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/>'s would be <img src="images/81f710835c45bfaf18fe1e7bb9a8939b.svg?invert_in_darkmode" align=middle width=40.79241539999999pt height=24.65753399999998pt/>, or the value of <img src="images/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> evaluated at each vertex of my tetrahedron. 

All of this means that a reasonable way to approximate any function in our tetrahedron is to use

<p align="center"><img src="images/478350d0d864b8692fe97555bff414b3.svg?invert_in_darkmode" align=middle width=138.01157369999999pt height=47.35857885pt/></p>

where <img src="images/2e26eb0900ac08bcaf445217a8686195.svg?invert_in_darkmode" align=middle width=35.29674884999999pt height=24.65753399999998pt/> are now the tetrahedron barycentric coordinates and <img src="images/9b6dbadab1b122f6d297345e9d3b8dd7.svg?invert_in_darkmode" align=middle width=12.69888674999999pt height=22.831056599999986pt/> are the values of <img src="images/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> at the nodes of the tetrahedron. Because our basis functions are linear, and the weighted sum of linear functions is still linear, this means that we are representing our function using a linear function space. 

## The Extension to 3D Movement

To apply this idea to physics-based animation of wiggly bunnies we need to more clearly define some of the terms above. First, we need to be specific about what our function <img src="images/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> will be. As with the particles in the previous assignments, what we care about tracking is the position of each mesh vertex, in the world, over time. For the <img src="images/c36ef0ba4721f49285945f33a25e7a45.svg?invert_in_darkmode" align=middle width=20.92202969999999pt height=26.085962100000025pt/> vertex we can denote this as <img src="images/07fa8f51f4cd0e31f63743c0638a3388.svg?invert_in_darkmode" align=middle width=56.34005519999999pt height=26.76175259999998pt/>. We are going to think of this value as a mapping from some undeformed space <img src="images/ae79425c6396a3fc7acd3d31d3bc3bc5.svg?invert_in_darkmode" align=middle width=54.86741699999999pt height=26.76175259999998pt/> into the real-world. So the function we want to approximate is <img src="images/0a6733943ebc99bb70de1c5cda4d14c5.svg?invert_in_darkmode" align=middle width=45.58205849999999pt height=26.085962100000025pt/> which, using the above, is given by

<p align="center"><img src="images/d39736c0e076aef46a625c5c707477b0.svg?invert_in_darkmode" align=middle width=154.83307785pt height=47.35857885pt/></p>

The take home message is that, because we evaluate <img src="images/c83e439282bef5aadf55a91521506c1a.svg?invert_in_darkmode" align=middle width=14.44544309999999pt height=22.831056599999986pt/>'s in the undeformed space, we need our tetrahedron to be embedded in this space. 

## The Generalized Coordinates 

Now that we have our discrete structure setup, we can start "turning the crank" to produce our physics simulator. A single tetrahedron has four (4) vertices. Each vertex has a single <img src="images/2982880f70a8a26b2a8893ae4db190b0.svg?invert_in_darkmode" align=middle width=14.942908199999989pt height=26.085962100000025pt/> associated with it. As was done in assignment 2, we can store these *nodal positions* as a stacked vector and use them as generalized coordinates, so we have

<p align="center"><img src="images/3ca0fe29482c50082a90537c92bcffc0.svg?invert_in_darkmode" align=middle width=77.84682345pt height=78.9048876pt/></p>

Now let's consider the velocity of a point in our tetrahedron. Given some specific <img src="images/d05b996d2c08252f77613c25205a0f04.svg?invert_in_darkmode" align=middle width=14.29216634999999pt height=22.55708729999998pt/>, the velocity at that point is

<p align="center"><img src="images/fcd1de7e861cf2d0a4a5d2a72aa12161.svg?invert_in_darkmode" align=middle width=186.53849279999997pt height=47.35857885pt/></p>

However, only the nodal variables actually move in time so we end up with

<p align="center"><img src="images/bb6065b2aa874eda6d41b8b2e1951471.svg?invert_in_darkmode" align=middle width=177.86275485pt height=47.35857885pt/></p>

Now we can rewrite this whole thing as a matrix vector product

<p align="center"><img src="images/fba0226e6f733da0926fe0f5f7a4b81e.svg?invert_in_darkmode" align=middle width=416.01854414999997pt height=103.61714219999999pt/></p>

where <img src="images/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.515988249999989pt height=22.465723500000017pt/> is the <img src="images/46e42d6ebfb1f8b50fe3a47153d01cd2.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> Identity matrix. 

## The Kinetic Energy of a Single Tetrahedron

Now that we have generalized coordinates and velocities we can start evaluating the energies required to perform physics simulation. The first and, and simplest energy to compute is the kinetic energy. The main difference between the kinetic energy of a mass-spring system and the kinetic energy of an FEM system, is that the FEM system must consider the kinetic energy of every infinitesimal piece of mass inside the tetrahedron. 

Let's call an infinitesimal chunk of volume <img src="images/7ccb42e2821b2a382a72de820aaec42f.svg?invert_in_darkmode" align=middle width=21.79800149999999pt height=22.831056599999986pt/>. If we know the density <img src="images/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode" align=middle width=8.49888434999999pt height=14.15524440000002pt/> of whatever our object is made out of, then the mass of that chunk is <img src="images/b393a71d7dbc7ba9bef53a45b498474d.svg?invert_in_darkmode" align=middle width=30.29688419999999pt height=22.831056599999986pt/> and the kinetic energy, <img src="images/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/> is <img src="images/e313402671b5400b40c904770c63159a.svg?invert_in_darkmode" align=middle width=140.34171855pt height=40.0576077pt/>. To compute the kinetic energy for the entire tetrahedron, we need to integrate over it's volume so we have

<p align="center"><img src="images/f20040204ebc8a72f628b655070e699b.svg?invert_in_darkmode" align=middle width=311.59084605pt height=37.3519608pt/></p>

BUT~ <img src="images/6877f647e069046a8b1d839a5a801c69.svg?invert_in_darkmode" align=middle width=9.97711604999999pt height=22.41366929999999pt/> is constant over the tetrahedron so we can pull that outside the integration leaving

<p align="center"><img src="images/63d88efc25d5db55efb8d46d2ca78bc2.svg?invert_in_darkmode" align=middle width=338.53143885pt height=63.53458154999999pt/></p>

in which the *per-element* mass matrix, <img src="images/f26134068b0b67870aaea679207afc4b.svg?invert_in_darkmode" align=middle width=22.18443314999999pt height=22.465723500000017pt/>, makes an appearance.. In the olden days, people did this integral by hand but now you can use symbolic math packages like *Mathematica*, *Maple* or even *Matlab* to compute its exact value. 

## The Deformation of a Single Tetrahedron

Now we need to define the potential energy of our tetrahedron. Like with the spring, we will need a way to measure the deformation of our tetrahedron. Since the definition of length isn't easy to apply for a volumetric object, we will try something else -- we will define a way to characterize the deformation of a small volume of space. Remember that all this work is done to approximate the function <img src="images/0a6733943ebc99bb70de1c5cda4d14c5.svg?invert_in_darkmode" align=middle width=45.58205849999999pt height=26.085962100000025pt/> which maps a point in the undeformed object space, <img src="images/d05b996d2c08252f77613c25205a0f04.svg?invert_in_darkmode" align=middle width=14.29216634999999pt height=22.55708729999998pt/>, to the world, or deformed space. Rather than consider what happens to a point under this mapping, let's consider what happens to a vector.  

To do that we pick two arbitary points in the undeformed that are infinitesimally close. We can call them <img src="images/5cb6ebd57b9c3ef9c5bf226aa856c60d.svg?invert_in_darkmode" align=middle width=20.84471234999999pt height=22.55708729999998pt/> and <img src="images/c89fe8937285e2a8ec46151e34cdff69.svg?invert_in_darkmode" align=middle width=20.84471234999999pt height=22.55708729999998pt/> (boring names I know). The vector between them is <img src="images/661cc435e9a81c8ee162d5ae09e11595.svg?invert_in_darkmode" align=middle width=107.98478954999997pt height=22.831056599999986pt/>. Similarly the vector between their deformed counterparts is <img src="images/266058ae232163eb4de3d7537b0fe02e.svg?invert_in_darkmode" align=middle width=154.29742514999998pt height=24.65753399999998pt/>. Because we chose the undeformed points to be infinitesimally close and <img src="images/f027e3c4319ee773dd95bd75d4e0408c.svg?invert_in_darkmode" align=middle width=159.81110475pt height=24.65753399999998pt/>, we can 
use Taylor expansion to arrive at

<p align="center"><img src="images/2a1f81d7a975db283cdc9e11ae4040d9.svg?invert_in_darkmode" align=middle width=119.45285385pt height=57.1636461pt/></p>

where <img src="images/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/> is called the deformation gradient. Remember, <img src="images/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/> results from differentiating a <img src="images/5dc642f297e291cfdde8982599601d7e.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/>-vector by another <img src="images/5dc642f297e291cfdde8982599601d7e.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/>-vector so it is a <img src="images/9f2b6b0a7f3d99fd3f396a1515926eb3.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> matrix.

Because <img src="images/439d5b12dc6c860995693aa45e4255d1.svg?invert_in_darkmode" align=middle width=23.46465164999999pt height=22.831056599999986pt/> is pointing in an arbitrary direction, <img src="images/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/>, captures information about how any <img src="images/439d5b12dc6c860995693aa45e4255d1.svg?invert_in_darkmode" align=middle width=23.46465164999999pt height=22.831056599999986pt/> changes locally, it encodes volumetric deformation. 

The FEM discretization provides us with a concrete formula for <img src="images/4ae122305a485974b9c14dcefe22cae8.svg?invert_in_darkmode" align=middle width=39.79437164999999pt height=24.65753399999998pt/> which can be differentiated to compute <img src="images/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/>. *An important thing to keep in mind --* because our particular FEM uses linear basis functions inside of a tetrahedron, the deformation gradient is a constant. Physically this means that all <img src="images/439d5b12dc6c860995693aa45e4255d1.svg?invert_in_darkmode" align=middle width=23.46465164999999pt height=22.831056599999986pt/>'s are deformed in exactly the same way inside a tetrahedron.

Given <img src="images/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/> we can consider the squared length of any <img src="images/74380e4b90b7786c87c490f3d94f2f68.svg?invert_in_darkmode" align=middle width=17.95095224999999pt height=22.831056599999986pt/> 

<p align="center"><img src="images/634073c68b4832cc0868f0293d1d432c.svg?invert_in_darkmode" align=middle width=406.03091759999995pt height=49.59096285pt/></p>

Like the spring strain, <img src="images/cbe99f908f3661c44ad20094523bbf90.svg?invert_in_darkmode" align=middle width=36.06344114999999pt height=27.6567522pt/> is invariant to rigid motion so it's a pretty good strain measure. 

## The Potential Energy of a Single Tetrahedron

The potential energy function of a tetrahedron is a function that associates a single number to each value of the deformation gradient. Sadly, for the FEM case, things are a little more complicated than just squaring <img src="images/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.85392569999999pt height=22.465723500000017pt/> (but thankfully not much). 

### The Strain Energy density
In this assignment we use the Stable Neohookean strain energy density from Theodore Kim's [Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/). The formula for this energy relies on computing what are called invariants of the Right Cauchy-Green Strain tensor. In this case we will define $I_2 = tr(\mathbf{F}^T\mathbf{F})$, where $\mathbf{F}\in\mathcal{}R^{3x3}$ is the deformation gradient and $tr()$ computes the trace of a matrix. We will also define $I_3 = det(\mathbf{F}$ which is just the determinant of the deformation gradient. The stable Neohookean potential energy is then given by:

$\Psi(\mathbf{F}) = \frac{1}{2}\mu(I_2 - 3) - \mu(I_3-1) + \frac{1}{2}(I_3 -1)^2$ 


### Numerical quadrature
Typically we don't evaluate potential energy integrals by hand. They get quite impossible, especially as the FEM basis becomes more complex. To avoid this we typically rely on [numerical quadrature](https://en.wikipedia.org/wiki/Numerical_integration). In numerical quadrature we replace an integral with a weighted sum over the domain. We pick some quadrature points <img src="images/9b01119ffd35fe6d8a8795a24fc11616.svg?invert_in_darkmode" align=middle width=18.943064249999992pt height=22.55708729999998pt/> (specified in barycentric coordinates for tetrahedral meshes) and weights <img src="images/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/> and evaluate

<p align="center"><img src="images/7f5d5d26767ddfc8af4bc7e06443c91a.svg?invert_in_darkmode" align=middle width=164.90166989999997pt height=48.1348461pt/></p>

However, for linear FEM, the quadrature rule is exceedingly simple. Recall that linear basis functions imply constant deformation per tetrahedron. That means the strain energy density function is constant over the tetrahedron. Thus the perfect quadrature rule is to choose <img src="images/9b01119ffd35fe6d8a8795a24fc11616.svg?invert_in_darkmode" align=middle width=18.943064249999992pt height=22.55708729999998pt/> as any point inside the tetrahedron (I typically use the centroid) and <img src="images/c2a29561d89e139b3c7bffe51570c3ce.svg?invert_in_darkmode" align=middle width=16.41940739999999pt height=14.15524440000002pt/> as the volume of the tetrahedron. This is called *single point* quadrature because it estimates the value of an integral by evaluating the integrated function at a single point. 

## Forces and stiffness

The per-element generalized forces acting on a single tetrahedron are given by 

<p align="center"><img src="images/0a4b8813bf5f5396689dcc60d64ab92e.svg?invert_in_darkmode" align=middle width=72.3931461pt height=37.0084374pt/></p>

and the stiffness is given by 

<p align="center"><img src="images/846606bb011963601357e7ffbb1c2e4d.svg?invert_in_darkmode" align=middle width=87.95259164999999pt height=38.973783749999996pt/></p>

These can be directly computed from the quadrature formula above. Again, typically one uses symbolic computer packages to take these derivatives and you are allows (and encouraged) to do that for this assignment. 

For a tetrahedron the per-element forces are a <img src="images/1035a4ac6789d259e5b11678e1db3967.svg?invert_in_darkmode" align=middle width=44.748820049999985pt height=21.18721440000001pt/> vector while the per-element stiffness matrix is a dense, <img src="images/97ab0bce852d417ba9871adc2af7bb26.svg?invert_in_darkmode" align=middle width=52.968029399999985pt height=21.18721440000001pt/> matrix. 

## From a Single Tetrahedron to a Mesh 
Extending all of the above to objects more complicated than a single tetrahedron is analogous to our previous jump from a single spring to a mass-spring system. 

![A tetrahedral mesh](images/tet_mesh.png)

The initial step is to divide the object to be simulated into a collection of tetrahedra. Neighboring tetrahedra share vertices. We now specify the generalized coordinates of this entire mesh as 

<p align="center"><img src="images/cc44391faa5e8aa1e9178dd46a4b72ae.svg?invert_in_darkmode" align=middle width=79.37236725pt height=108.49566915pt/></p> 

where <img src="images/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the number of vertices in the mesh. We use selection matrices ) which yield identical assembly operations for the global forces, stiffness and mass matrix. In this assignment the assembly code is given to you in the [given](./given) subdirectory. Feel free to take a look at [it](./given/mass_matrix_object.py)

## Some Debugging Hints
1. Always test the one_tet_* examples first, this rules out anyting going wrong with assembly or other global operations
2. Check results in order of complexity of simulation, so start with *_stationary, then move onto *_falling, then *_deflate and finally *_swinging. 
3. If you are using Visual Studio Code or Cursor, use the [interactive debugger](https://code.visualstudio.com/docs/python/debugging) and python debugging console.
   
## Admissable Code and Libraries
You are allowed to use SymPy for computing formulas for integrals, derivatives and gradients. You are allowed to use any functions in the warp and warp.sparse packages. You ARE NOT allowed to use code from other warp packages like warp.fem. You are not allowed to use any of warps specialized spatial data structures for storing meshes, volumes or doing spatial subdivision. You cannot use code from any other external simulation library.  

## Hand-In
We will collect and grade the assignment using MarkUs, link **coming soon**

## Late Penalty
The late penalty is the same as for the course, specified on the main github page. 
