### @file   init_cond.py
### @author Chandler Ross
### @date   March 17, 2020
### @brief  Module for reading particle position and velocity initial conditions from file.

### @brief  Fills arrays with the initial conditions given in a file. (in a non memory coalesced way)
### @param     r      A 2D array: 1st dimension is the number of particles, 2nd is their positions in 3D space.
### @param     v      A 2D array: 1st dimension is the number of particles, 2nd is their velocities in 3D space.
### @param     m      A 1D array: dimension is the number of particles, each element contains that particles mass.
### @param  fileName  The name of the file to read data from.  
import numpy as np

def initial_Conditions(r, v, m, fileName):
    
	File = open(fileName,"r")
	lines = File.readlines()
	numParticles = len(lines)-1
    
	for i in np.arange(numParticles+1)[1:]:  # read from the file in such a way that we can have memory coalescence
		info = lines[i].split()

		for j in np.arange(7):
			if j == 0:
				m[i-1] = float(info[j])
			elif j == 1:
				r[i-1] = float(info[j])
			elif j == 2:
				r[i-1+numParticles] = float(info[j])
			elif j == 3:
				r[i-1+2*numParticles] = float(info[j])
			elif j == 4:
				v[i-1] = float(info[j])
			elif j == 5:
				v[i-1+numParticles] = float(info[j])
			elif j == 6:
				v[i-1+2*numParticles] = float(info[j])

	"""for i in np.arange(len(lines))[1:]:
		info = lines[i].split()
		m[i-1]   = float(info[0])
		r[3*i-3] = float(info[1])
		r[3*i-2] = float(info[2])
		r[3*i-1] = float(info[3]) 
		v[3*i-3] = float(info[4])
		v[3*i-2] = float(info[5])
		v[3*i-1] = float(info[6])"""

	File.close()
    
	return r, v, m
