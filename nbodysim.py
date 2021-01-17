### @file   nbodysim.py
### @author Chandler Ross
### @date   March 17, 2020
### @brief  The main driver file to execute code from all the modules in this directory for the N body simulation
import ctypes
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import math as M
import timeit

from init_cond import initial_Conditions
from runPlot import runPlot
from runError import runError

# Parameters for simulation
# Redefine units such that mass of Jupiter (M) = 1 and G = 1
G     		  = 6.673e-11                   	# gravitational constant
M0    		  = 1.898e27                    	# set mass scale
R0    		  = 8.8605e9                    	# set length scale
T0    	 	  = np.sqrt(R0**3/(G * M0))     	# set time scale
rH            = 5.37e10/R0                      # Hill radius (scaled)
flag 		  = "-p"							# decide what part of program to execute... -p = plot, -e = error			
dt 			  = 86400/T0						# default time step (arbitrary)
n 			  = 1								# Lowers the time step for each call to A1 and A2. Also more calls
numSteps 	  = 2     						# default number of time steps to take (arbitrary)
fileName 	  = "particles.txt"  			 	# file to read initial conditions from
File 		  = open(fileName, "r")
lines 		  = File.readlines()
numParticles  = len(lines) - 1 			       	# number of particles in simulation
File.close()
numParticles  = 1024
r 			  = np.zeros(3 * numParticles)		# array to hold positions of particles
v 			  = np.zeros(3 * numParticles)  	# array to hold velocities of particles
m 			  = np.zeros(numParticles)	        # array to hold masses of particles
dist		  = np.zeros(numParticles)
dirvec 		  = np.zeros(3)						# array to find direction vector along particle j to particle i
timeStep_iter = np.logspace(-4,-1,13)           # loop over time steps
#runTime 	  = np.zeros(len(timeStep_iter))    # the total run time
rel_err 	  = np.zeros(len(timeStep_iter))    # largest relative error
#runTimeLF 	  = np.zeros(len(timeStep_iter))   	# the total run time for LF
#rel_errLF 	  = np.zeros(len(timeStep_iter))    # largest relative error for LF
eps 		  = 0.001							# softening factor
ecc			  = np.zeros(numParticles)			# eccentricity vector
status 		  = np.ones(numParticles)			# status vector for collisions and ejections
rSatellites   = np.array([1e5/R0, 1e5/R0])

r, v, m = initial_Conditions(r, v, m, fileName)

if flag == "-p":
	r, v, m = initial_Conditions(r, v, m, fileName)
	runPlot(r, v ,m, numSteps, numParticles, dt, n, rSatellites, rH, status, eps)
elif flag == "-e":
	# make error and run time plot
	runTime, rel_err = runError(r, v, m, numParticles, n, dt, numSteps, ecc, status, eps, rSatellites)

	plt.figure(2)
	plt.loglog(timeStep_iter,rel_err)
	plt.title('')
	plt.xlabel('Time Step')
	plt.ylabel('Relative Error')

"""sim  = ctypes.CDLL('./runSim.so')
#sim2 = ctypes.CDLL('./runSim2.so')
sim.runSim(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 			    \
           m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt), ctypes.c_int(numParticles),  			    \
		   ctypes.c_int(n), ctypes.c_double(eps), ctypes.c_int(numSteps), ecc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  \
		   status.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rSatellites.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
		   dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

distReal = [i/rH for i in dist]

for i in np.arange(numParticles)[1:]:
	plt.figure(num=1, figsize=(9, 3), dpi=200, facecolor='w', edgecolor='k')
	if i == 1:
		plt.plot(distReal[1:], [0 for i in distReal][1:], c='lightskyblue', linewidth=0.7, zorder=1)
		plt.scatter(distReal[i], ecc[i], s=20, c='red', edgecolor='k', zorder=4)
	elif i%2==0:
		plt.scatter(distReal[i], ecc[i], s=5, c='mediumturquoise',zorder=i%3)
	elif i%2==1:
		plt.scatter(distReal[i], ecc[i], s=5, c='sandybrown',zorder=i%3)
	plt.xlabel(r'r ($r_{Hill}$)', fontsize=14)
	plt.ylabel('e', fontsize=14)
	plt.text(0.21, 0.35, 't = 0 years', fontsize=14)
	plt.xticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
	plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
plt.show()"""
#plt.savefig('distrib{}.png'.format(numSteps))

"""sim2.runSim2(ctypes.c_double(dt), ctypes.c_int(numParticles), ctypes.c_int(n), ctypes.c_double(eps), ctypes.c_int(numSteps), \
		     ecc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), status.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),    \
		     rSatellites.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))"""

