### @file   runPlot.py
### @author Chandler Ross
### @date   March 19, 2020
### @brief  Plots the output from the N body simulation.

import ctypes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

sim   = ctypes.CDLL('./runSim2.so')

### @brief Module computes the method and plots the output.
### @param      r         A 1D array: Lists the x,y,z position of particle 0, then 1, ...
### @param      v         A 1D array: Lists the vx,vy,vz position of particle 0, then 1, ...
### @param      m         A 1D array: contains the masses for particle 0, 1, ..., N-1.
### @param    numSteps    Integer > 0... The number of times the loop iterates. Sets how long the simulation runs.
### @param  numParticles  The number of particles ie. the size of the m array.
### @param      dt        The time step over which you wish to update the positions.
### @param      n         Integer > 0... Lower the timestep and how many times you call A1 and A2.
def runPlot(r, v, m, numSteps, numParticles, dt, n, rSatellites, rH, status, eps, ecc):
	# Store the updated values
	# Format: Rx = [x01,x11,...,xN1,x02,x12,...,xN2,...]
	# First digit is the particle, second is the time step
	Rx = np.zeros(numSteps*numParticles)
	Ry = np.zeros(numSteps*numParticles)
	Rz = np.zeros(numSteps*numParticles)

	for i in np.arange(numSteps):
		sim.runSim2(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                    m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt), ctypes.c_int(numParticles), ctypes.c_int(n), ctypes.c_double(eps), ctypes.c_int(1), \
          	        ecc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), status.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),    \
            	    rSatellites.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

		for j in np.arange(numParticles):
			# x,y and z components of each planet
			# for each time step
			Rx[numParticles*i+j] = r[j]
			Ry[numParticles*i+j] = r[j+numParticles]
			Rz[numParticles*i+j] = r[j+2*numParticles]

		#sim.runSim(r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
         #  							m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_double(dt), ctypes.c_uint(numParticles),  \
          # 							ctypes.c_uint(n), ctypes.c_double(eps))		

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	for i in np.arange(numParticles):
		if i == 0:
			ax.plot(Rx[i::numParticles],Ry[i::numParticles],Rz[i::numParticles], c='magenta')
		elif i == 1:
			ax.plot(Rx[i::numParticles],Ry[i::numParticles],Rz[i::numParticles], c='cyan')
		else:
			ax.plot(Rx[i::numParticles],Ry[i::numParticles],Rz[i::numParticles], c='black')
	plt.title("Real Space N Body Problem: HR")
	ax.set_zlim(-0.001, 0.001)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	plt.show()
