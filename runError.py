### @file   runError.py
### @author Chandler Ross
### @date   March 19, 2020
### @brief  Returns arrays to plot the error for the method as well as the run time.

import ctypes
import math as M
import numpy as np
import matplotlib.pyplot as plt
import timeit

nrg   					 = ctypes.CDLL('./energy.so')
nrg.energy.restype 		 = ctypes.c_double  					# so that it returns a double
nrg.energynew.restype 	 = ctypes.c_double
nrg.angMomentum.restype  = ctypes.c_double
nrg.linMomentum.restype  = ctypes.c_double
nrg.totalMass.restype    = ctypes.c_double
nrg.crossProduct.restype = ctypes.c_double

from init_cond import initial_Conditions
sim   = ctypes.CDLL('./runSim.so')

# parameters
time 		  = 1												# total time to run for each of the time steps
dirvec 		  = np.zeros(3)
timeStep_iter = np.logspace(-4,-1,13)							# loop over time steps
numSteps 	  = np.array([time/i for i in timeStep_iter])		# number of steps to reach the total time
rel_err 	  = np.zeros(len(timeStep_iter))					# largest relative error
start 		  = np.zeros(len(timeStep_iter))					# for where we start the run time clock
stop 		  = np.zeros(len(timeStep_iter))					# for where we end the run time clock
runTime 	  = np.zeros(len(timeStep_iter))					# the total run time
fileName 	  = "particles.txt"                          		# file to read initial conditions from
rHembryo 	  = 0.006244996
#eps 		  = [(rHembryo/20)*i for i in np.arange(21)]

### @brief Module computes the error and run time and returns arrays to plot the output.
### @param      r         A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
### @param      v         A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
### @param      m         A 1D array: contains the masses for particle 0, 1, ..., N-1.
### @param    numSteps    Integer > 0... The number of times the loop iterates. Sets how long the simulation runs.
### @param  numParticles  The number of particles ie. the size of the first index of r and v.
### @param      n         Integer > 0... Lower the timestep and how many times you call A1 and A2.
def runError(r, v, m, numParticles, n, dt, numSteps, ecc, status, eps, rSatellites):

	for k in eps:	

		r, v, m = initial_Conditions(r, v, m, fileName)

		# initial energy
		E0 = nrg.energynew(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
						   m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(numParticles), ctypes.c_double(k)) 

		for i in np.arange(len(timeStep_iter)):
        
			# Holds relative error for each time step
			rel_err_iter = np.zeros(int(M.ceil(numSteps[i])))
                                
			r, v, m = initial_Conditions(r, v, m, fileName)
	
			#start[i] = timeit.default_timer()

			for j in np.arange(int(M.ceil(numSteps[i]))):
        
				sim.runSim(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
           				   m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(timeStep_iter[i]), ctypes.c_uint(numParticles),  \
             			   ctypes.c_uint(n), ctypes.c_double(k))			

				E = nrg.energynew(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                	              m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(numParticles), ctypes.c_double(k))
			
				rel_err_iter[j] = abs((E0 - E) / E0)

			#stop[i] = timeit.default_timer()
			#runTime[i] = stop[i] - start[i]
    
			rel_err[i] = max(rel_err_iter)
		
		print(k)
		"""plt.figure(2)
		plt.loglog(timeStep_iter,rel_err, label='{}'.format(k))
		plt.xlabel('Time Step')
		plt.ylabel('Relative Error')"""

	#plt.legend(loc='best')
	#plt.show()

	"""M_iter = np.zeros(numSteps)
	L_iter = np.zeros(numSteps)
	P_iter = np.zeros(numSteps)
	E_iter = np.zeros(numSteps)

	for i in np.arange(numSteps):
		M_iter[i] = nrg.totalMass(m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(numParticles))
		L_iter[i] = nrg.angMomentum(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                    m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(numParticles))
		P_iter[i] = nrg.linMomentum(v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                    ctypes.c_uint(numParticles))
		E_iter[i] = nrg.energynew(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),   \
                                  m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint(numParticles), ctypes.c_double(eps))
	
		sim.runSim(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 			    \
                   m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt), ctypes.c_uint(numParticles),  			    \
                   ctypes.c_uint(n), ctypes.c_double(eps), ctypes.c_uint(numSteps), ecc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
				   status.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rSatellites.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))"""

	"""plt.figure(0)
	plt.plot(np.arange(numSteps), M_iter)
	plt.xlabel('Days')
	plt.ylabel('Total Mass')

	plt.figure(1)
	plt.plot(np.arange(numSteps), L_iter)
	plt.xlabel('Days')
	plt.ylabel('Total Angular Momentum')

	plt.figure(2)
	plt.plot(np.arange(numSteps), P_iter)
	plt.xlabel('Days')
	plt.ylabel('Total Linear Momentum')"""

	"""plt.figure(3)
	plt.plot(np.arange(numSteps), E_iter)
	plt.xlabel('Days')
	plt.ylabel('Total Energy')

	plt.show()"""

	return runTime, rel_err
