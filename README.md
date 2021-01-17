# NbodySimGPU
Compute the symplectic integrator method described by Hanno Rein in Embedded operator splitting methods for perturbed systems optimized for the GPU.
The code should be extensively commented.

Unfortunately, pinned memory is not being utilized in this simulation.

runSim.cu and nbodysim.py   are the uncoalesced versions of the simulation
runSim2.cu and nbodysim2.py are the coalesced versions of the simulation
createIC.py 				initializes the mass, velocity, and position of each particle
init_cond.py 				reads the initial conditions and stores them. Not needed in the coalesced version
load.sh 					loads the required modules
makefile 					used to compile. Just type 'make'. There is also a 'make clean' option
particles.py 				is where the data from createIC.py is sent.
runPlot.py					makes a plot of all the particles' evolution

To compile, type "make"
To run, type "python nbodysim2.py"

The collision function in runSim2.py doesn't really work
