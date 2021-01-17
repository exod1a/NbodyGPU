import numpy as np
import random
import math as m

""" Initialize particle masses, positions and velocities for simulation.
	1000 particles reside within a disk between 0.03rH and 0.3rH where
	rH is the Hill radius. They have an average eccentricity (e) of 1e-3
	and average inclination (i) of 1e-3. They have a radius of 100 km. 
	The Hill radius is 0.359 AU i.e. 5.37e10 m.
	The total mass of the disk is 2e-7M. where M. = 1.989e30 kg (mass of Sun)
	Therefore, each of the super satelitesimals has a mass of 3.987e20 kg.
	Mo is the mass of the central planet and is meant to be 1e-3M. i.e.
	1.989e27 kg. Although the actual mass of Jupiter is 1.898e27 kg. 
	The satelitesimals don't interact with each other, only the central planet
	and the satellite embryo. As far as I can tell, the embryo has the same 
	initial mass as the satelitesimals but it accretes mass. 
	The central planet will be at 0,0,0 while the embryo will start at
	0.25rH with a very small eccentricity. 
"""

# Initialize particle x, y and z positions in the disk
#def setICs():

numParticles = m.floor(1024)

# Redefine units such that mass of Jupiter (M) = 1 and G = 1
G     = 6.673e-11					# gravitational constant
M0    = 1.898e27					# set mass scale
R0    = 8.8605e9					# set length scale
T0    = np.sqrt(R0**3/(G * M0))		# set time scale

# Simulation parameters
rH    = 5.37e10 					# Hill radius (not scaled)
r_in  = 0.03*rH	    	  			# inner bound of satelitesimal disk (not scaled)
r_out = 0.3*rH						# outer bound of satelitesimal disk (not scaled)
m_one = 3.987e20/M0	     			# mass of satelitesimals
M     = 1.898e27/M0		    		# mass of Jupiter (central planet)
"""m     = np.array([m_one for i in np.arange(1023)])
r     = np.array([[0., 0., 0.] for i in np.arange(1023)])
v     = np.array([[0., 0., 0.] for i in np.arange(1023)])

# random inclination with average of 0
inc = np.random.normal(0, 1e-3, 1023)		

# generate random semi-major axes within disc
a = np.zeros(1023)

# set eccentricities
e = np.random.normal(5e-2, 2e-2, 1023)    #ask Hanno
for i in np.arange(len(a)):
	if e[i] < 0:
		e[i] = -e[i]
	a[i] = random.randint(r_in, r_out)/R0

for i in np.arange(len(a)):
	# initialize at pericenter where y = 0, Vx = 0
	r[i][0] = a[i] * (1 - e[i])

	# use energy equation to solve for Vy
	v[i][1] = np.sqrt(2 * (m[i] + M) * (1 / r[i][0] - 1./(2 * a[i])))

	# use for random rotation around z-axis
	theta   = np.radians(random.randint(0, 360))
	c, s    = np.cos(theta), np.sin(theta)
	rotateZ = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

	# use for random rotation around x-axis for inclination
	c2, s2  = np.cos(np.radians(inc[i])), np.sin(np.radians(inc[i]))
	rotateX = np.array(((1, 0, 0), (0, c2, -s2), (0, s2, c2)))

	# perform rotations
	r[i]    = np.matmul(rotateZ, r[i])
	v[i]	= np.matmul(rotateZ, v[i])
	r[i]    = np.matmul(rotateX, r[i])
	v[i]    = np.matmul(rotateX, v[i])

# fix the first particle in the array (embryo) so
# it is at the proper starting position (not random)
a[0]    = 0.25*rH/((1-e[0])*R0)
r[0][0] = 0.25*rH/R0
r[0][1], r[0][2] = 0, 0
v[0][1] = np.sqrt(2 * (m[0] + M) * (1 / r[0][0] - 1./(2 * a[0])))
v[0][0], v[0][2] = 0, 0

# use for test case purposes
a[0] = (0.03*rH/R0-0.01)/(1-e[0])
r[0][0] = a[0]*(1+e[0])
r[0][1], r[0][2] = 0, 0
v[0][1] = np.sqrt(2 * (m[0] + M) * (1 / r[0][0] - 1./(2 * a[0])))
v[0][0], v[0][2] = 0, 0

# print results to screen to dump into "particles.txt" file
print(" ")
print(M, "0 0 0 0 0 0")
for i in np.arange(len(a)):
	print(m[i], r[i][0], r[i][1], r[i][2], v[i][0], v[i][1], v[i][2])

print(e[0])"""

m     = np.array([m_one for i in np.arange(numParticles)])
r     = np.array([[0., 0., 0.] for i in np.arange(numParticles)])
v     = np.array([[0., 0., 0.] for i in np.arange(numParticles)])

# random inclination with average of 0
inc = np.random.normal(0, 1e-3, numParticles)

# generate random semi-major axes within disc
a = np.zeros(numParticles)

# set eccentricities
e = np.random.normal(5e-2, 2e-2, numParticles)    #ask Hanno
for i in np.arange(len(a)):
    if e[i] < 0:
        e[i] = -e[i]
    a[i] = random.randint(r_in, r_out)/R0

for i in np.arange(len(a)):
    # initialize at pericenter where y = 0, Vx = 0
    r[i][0] = a[i] * (1 - e[i])

    # use energy equation to solve for Vy
    v[i][1] = np.sqrt(2 * (m[i] + M) * (1 / r[i][0] - 1./(2 * a[i])))

    # use for random rotation around z-axis
    theta   = np.radians(random.randint(0, 360))
    c, s    = np.cos(theta), np.sin(theta)
    rotateZ = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

    # use for random rotation around x-axis for inclination
    c2, s2  = np.cos(np.radians(inc[i])), np.sin(np.radians(inc[i]))
    rotateX = np.array(((1, 0, 0), (0, c2, -s2), (0, s2, c2)))

    # perform rotations
    r[i]    = np.matmul(rotateZ, r[i])
    v[i]    = np.matmul(rotateZ, v[i])
    r[i]    = np.matmul(rotateX, r[i])
    v[i]    = np.matmul(rotateX, v[i])

# fix the first particle in the array (embryo) so
# it is at the proper starting position (not random)
a[0]    = 0.25*rH/((1)*R0)
r[0][0] = 0.25*rH/R0
r[0][1], r[0][2] = 0, 0
v[0][1] = np.sqrt(2 * (m[0] + M) * (1 / r[0][0] - 1./(2 * a[0])))
v[0][0], v[0][2] = 0, 0

# print results to screen to dump into "particles.txt" file
print(" ")
print(M, "0 0 0 0 0 0")
for i in np.arange(len(a)):
    print(m[i], r[i][0], r[i][1], r[i][2], v[i][0], v[i][1], v[i][2])
