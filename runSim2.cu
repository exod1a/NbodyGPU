// runSim2.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <assert.h>

// Executes the A1 operator optimized

/// @brief Executes the A1 step of the algorithm. Updates the positions of the particles according to their velocities.

/// @variable r:  A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v:  A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable dt: The time-step	used for the simulation
__global__ void A1_kernel(double* r, double* v, double dt) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    r[id] += v[id] * dt; // update positions
}

// Coalesced memory A2

/// @brief Executes the A2 step of the algorithm. Updates the velocites of all particles due to the force from the
///		   central planet (particle 0) and the velocity of the central planet due to the forces from the other particles.

/// @variable r: 			A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v: 			A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m: 			A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable dt: 			The time-step used for the simulation
/// @variable varr: 		A 1D array of size 3n used to store the effects on particle 0 due to the others for the reduction routine
/// @variable status: 		A 1D array of size n originally initiated to all 1's, later changed to 0's if a particle is absorbed or ejected
/// @variable numParticles: The number of particles in the simulation: n
__global__ void A2_kernel(double *r, double *v, double *m, double dt, double *varr, double *status, int numParticles) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1;    // Starts at 1
    double invdist;
    double dirvec[3];

    if (id < numParticles) {
		// Vector that points along particle id to particle 0
        dirvec[0] = r[0]              - r[id];
        dirvec[1] = r[numParticles]   - r[numParticles+id];
        dirvec[2] = r[2*numParticles] - r[2*numParticles+id];

        // Distance between particle 0 and id
        invdist = dt * rnorm3d(dirvec[0], dirvec[1], dirvec[2])*\
                       rnorm3d(dirvec[0], dirvec[1], dirvec[2])*\
                       rnorm3d(dirvec[0], dirvec[1], dirvec[2]);

		// If particle's status has previously been set to 0, it has no effect
        if (status[id] == 0) {
			// Particle id remains at 0
            v[id]   	         += 0;
            v[numParticles+id]   += 0;
            v[2*numParticles+id] += 0;

			// Particle 0 has no interaction
            varr[id]                = 0;
            varr[numParticles+id]   = 0;
            varr[2*numParticles+id] = 0;
        }
        else {
            // Update velocities of particles 1 through N-1
            v[id]   		 	 += m[0] * invdist * dirvec[0];
            v[numParticles+id]   += m[0] * invdist * dirvec[1];
            v[2*numParticles+id] += m[0] * invdist * dirvec[2];

			// Store forces on particle 0
            varr[id]                = -m[id] * invdist * dirvec[0];
            varr[numParticles+id]   = -m[id] * invdist * dirvec[1];
            varr[2*numParticles+id] = -m[id] * invdist * dirvec[2];
        }

		// Unfortunately, this is executed by each thread but there is no race condition
		// Adds the previous values of particle 0's velocity to varr to include them in the reduction
        varr[0]              = v[0];
        varr[numParticles]   = v[numParticles];
        varr[2*numParticles] = v[2*numParticles];
    }
}

// Coalesced B operator

/// @brief Executes the B step of the algorithm. Updates the velocity of the planetary embryo due to the forces of the 
///		   the particles (other than the central planet). Normally, it would calculate the effect of all the other
///		   inter-particle interactions but those effects are neglected in this simulation.

/// @variable r: 			A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v: 			A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m: 			A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable dt: 			The time-step used for the simulation
/// @variable varr: 		A 1D array of size 3n used to store the effects on particle 0 due to the others for the reduction routine
/// @variable status: 		A 1D array of size n originally initiated to all 1's, later changed to 0's if a particle is absorbed or ejected
/// @variable numParticles: The number of particles in the simulation: n
/// @variable eps:			The gravitational softening parameter
__global__ void B_kernel(double *r, double *v, double *m, double *varr, double dt, int numParticles, double *status, double eps) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x + 2; // Starts at 2
    double dirvec[3];

    double invdist;

    if (id < numParticles) {
		// Vector that points along particle id to particle 0
        dirvec[0] = r[1]   			    - r[id];
        dirvec[1] = r[numParticles+1]   - r[numParticles+id];
        dirvec[2] = r[2*numParticles+1] - r[2*numParticles+id];

		// Distance between particle 0 and id
        invdist = status[id] * dt * rsqrt((dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps*eps)*\
                                          (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps*eps)*\
                                          (dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] + eps*eps));

        // Update id'th satellitesimal
        v[id]   			 += m[1] * invdist * dirvec[0];
        v[numParticles+id]   += m[1] * invdist * dirvec[1];
        v[2*numParticles+id] += m[1] * invdist * dirvec[2];

        // Update embryo
        // Store forces on embryo for reduction
        varr[0]                = v[1];
        varr[numParticles-1]   =    0;  			// These 0s are padding for the reduction
        varr[numParticles]     = v[numParticles+1];
        varr[2*numParticles-1] =    0;
        varr[2*numParticles]   = v[2*numParticles+1];
        varr[3*numParticles-1] =    0;

        varr[id-1]                = -m[id] * invdist * dirvec[0];
        varr[numParticles+id-1]   = -m[id] * invdist * dirvec[1];
        varr[2*numParticles+id-1] = -m[id] * invdist * dirvec[2];
    }
}

// Coalesced merge and eject

/// @brief Check if particle is still within the satellitesimal disc. If a particle is below 0.03*rH, it merges with the central planet.
///		   If it is above rH, it is ejected from the disc.

/// @variable r:            A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable status:       A 1D array of size n originally initiated to all 1's, later changed to 0's if a particle is absorbed or ejected
/// @variable numParticles: The number of particles in the simulation: n
/// @variable rH:			The Hill radius of of particle 0 (Jupiter)
__global__ void mergeEject(double *r, double *status, int numParticles, double rH) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x + 2; // Starts at 2
    double dist;

    if (id < numParticles) {
        dist = norm3d(r[0]-r[id], r[numParticles]-r[numParticles+id], r[2*numParticles]-r[2*numParticles+id]);

        if (dist < 0.03*rH && status[id] != 0)
            status[id] = 2;
        else if (dist > rH && status[id] != 0)
            status[id] = 3;  // so that momentum conservation doesn't include ejected particles
                             // will be set to 0 in the consMomentum function
    }
}

/// @brief If a particle merges with the central planet or a collision occurs with the embryo, momentum conservation
///  	   is needed to find the resulting velocity of the central planet or embryo (the other particle no longer exists).

/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable status:       A 1D array of size n originally initiated to all 1's, later changed to 0's if a particle is absorbed or ejected
/// @variable numParticles: The number of particles in the simulation: n
/// @variable rSatellites:  A 1D array of size 2 where the 0th entry is the embryo radius and the 1st entry is the radius of the satellitesimals 
__global__ void consMomentum(double *v, double *m, double *status, int numParticles, double *rSatellites) {
    for (int id = 2; id < numParticles; id++) {
        if (status[id] == 2) {
            status[id]          = 0;
            // use conservation of momentum to update central velocity
            v[0]                = 1./(m[0] + m[id]) * (m[0]*v[0] + m[id]*v[id]);
            v[numParticles]     = 1./(m[0] + m[id]) * (m[0]*v[numParticles] + m[id]*v[numParticles+id]);
            v[2*numParticles]   = 1./(m[0] + m[id]) * (m[0]*v[2*numParticles] + m[id]*v[2*numParticles+id]);
            // conservation of mass
            m[0]               += m[id];
        }
		else if (status[id] == 4) {
            status[id]          = 0;
            rSatellites[0]      = cbrt((m[1]+m[2])/m[2])*rSatellites[1];
            // use conservation of momentum to update velocity
            v[1]                = 1./(m[1] + m[id]) * (m[1]*v[1] + m[id]*v[id]);
            v[numParticles+1]   = 1./(m[1] + m[id]) * (m[1]*v[numParticles+1] + m[id]*v[numParticles+id]);
            v[2*numParticles+1] = 1./(m[1] + m[id]) * (m[1]*v[2*numParticles+1] + m[id]*v[2*numParticles+id]);
            // conservation of mass
            m[1]               += m[id];
        }
		else if (status[id] == 3)
            status[id] = 0;
        else
            continue;
    }
}

/// @brief Each particle has a status initially set to 1. If a particle merges with the central planet or collides with
///		   the embryo, its status is set to 0 so that it no longer interacts in the simulation. This sets its mass, position
///		   and velocity to 0. 

/// @variable r:            A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable status:       A 1D array of size n originally initiated to all 1's, later changed to 0's if a particle is absorbed or ejected
/// @variable numParticles: The number of particles in the simulation: n
__global__ void statusUpdate(double *r, double *v, double *m, double *status, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    m[id/3] *= status[id/3];
   	r[id] 	*= status[id % numParticles];
    v[id] 	*= status[id % numParticles];
}

/// @brief Function to find cross product of two 3-vectors, vect_A and vect_B, and stores the result in cross_P (on the device)

/// @variable vect_A:  Any 3-vector
/// @variable vect_B:  Any 3-vector
/// @variable cross_P: Any 3-vector, holds the result of the cross product
__device__ void crossProduct(double *vect_A, double *vect_B, double *cross_P) {
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
}

/// @brief Function used for resolving collisiions with the satellitesimal and the embryo. First goes to the rest frame of the embryo,
/// 	   then evolves each particle according to the A1 function while the embryo remains fixed. If the distance between the embryo
///		   and the line traced out by the other particles is less than their combined radii, a collision occurs. 

/// @variable r:            A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable status:       A 1D array of size n originally initiated to all 1's, later changed to 0's if a particle is absorbed or ejected
/// @variable rSatellites:  A 1D array of size 2 where the 0th entry is the embryo radius and the 1st entry is the radius of the satellitesimals
/// @variable numParticles: The number of particles in the simulation: n
/// @variable dt:           The time-step used for the simulation
__global__ void collision(double* r, double* v, double* status, double* rSatellites, int numParticles, double dt) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 2;

    double rTemp[3]; 
    double vTemp[3];
    double crossP[3];
    double vecA[3];
    double vecB[3];
    double t;
    double dist;
    double d1;
    double d2;

	if (id < numParticles) {
    	// go to rest frame of embryo
    	vTemp[0] = v[id]   				- v[1];
    	vTemp[1] = v[numParticles+id]   - v[numParticles+1];
    	vTemp[2] = v[2*numParticles+id] - v[2*numParticles+1];

    	// evolve satelitesimal
    	rTemp[0] = r[id]   				+ vTemp[0] * dt/4.;
    	rTemp[1] = r[numParticles+id]   + vTemp[1] * dt/4.;
    	rTemp[2] = r[2*numParticles+id] + vTemp[2] * dt/4.;

    	// the equation ((r-r[1]) * (rTemp-r)) / |rTemp-r|^2 where r[1] is the embryo's
    	// position in its rest frame, r is the satelitesimal's original position and rTemp is the
    	// satelitesimal's updated position in the rest frame. * indicates a dot product in this case
    	// this is the time that minimizes the distance function from a line segment to a point
    	t = ((r[id]-r[1])							   *(rTemp[0]-r[id])       			 +\
         	 (r[numParticles+id]-r[numParticles+1])	   *(rTemp[1]-r[numParticles+id])    +\
         	 (r[2*numParticles+id]-r[2*numParticles+1])*(rTemp[2]-r[2*numParticles+id])) /\
        	((rTemp[0]-r[id])						   *(rTemp[0]-r[id])      			 +\
         	 (rTemp[1]-r[numParticles+id])			   *(rTemp[1]-r[numParticles+id])  	 +\
         	 (rTemp[2]-r[2*numParticles+id])		   *(rTemp[2]-r[2*numParticles+id]));

    	if (0 < t < 1) {
    		// the equation |(r[1]-r) x (r[1]-rTemp)|/|rTemp-r| where r[1] is the embryo's position
    		// in its rest frame, r is the satelitesimal's original position and rTemp is the
    		// satelitesimal's updated position in the rest frame
    		// if t is in this range, then the point in within line segment
 			vecA[0] = r[1]-r[id],    vecA[1] = r[numParticles+1]-r[numParticles+id], vecA[2] = r[2*numParticles+1]-r[2*numParticles+id];
			vecB[0]	= r[1]-rTemp[0], vecB[1] = r[numParticles+1]-rTemp[1],  		 vecB[2] = r[2*numParticles+1]-rTemp[2];    	
			crossProduct(vecA, vecB, crossP);
			dist 	= norm3d(crossP[0],crossP[1],crossP[2])*rnorm3d(rTemp[0]-r[id], rTemp[1]-r[numParticles+id], rTemp[2]-r[2*numParticles+id]);
    	}

    	else if (t > 1 || t < 0) {
    		// if t is not in the range, it does not lie within the line segment
    		// the equation |r-r[1]|
    		d1   = norm3d(r[id]-r[1], r[numParticles+id]-r[numParticles+1], r[2*numParticles+id]-r[2*numParticles+1]);

    		// the equation |rTemp-r[1]|
        	d2   = norm3d(rTemp[0]-r[1], rTemp[1]-r[numParticles+1], rTemp[2]-r[2*numParticles+1]);

			dist = fmin(d1, d2); 
    	}

		if (dist < rSatellites[0] + rSatellites[1])
			status[id] = 4;
	}
}

/// @brief Function to find cross product of two 3-vectors, vect_A and vect_B, and stores the result in cross_P

/// @variable vect_A:  Any 3-vector
/// @variable vect_B:  Any 3-vector
/// @variable cross_P: Any 3-vector, holds the result of the cross product
void crossProduct2(double *vect_A, double *vect_B, double *cross_P) {
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
}

/// @brief Finds the eccentricity of all particles

/// @variable r:            A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable ecc:			A 1D array of size n that holds the eccentricity of each particle
/// @variable numParticles: The number of particles in the simulation: n
__global__ void calcEccentricity(double *r, double *v, double *m, double *ecc, int numParticles) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x + 1;
	double L[3];                                                            // angular momentum
	double eccTemp[3];                                                      // hold components of eccentricity vector
	double mu;          					                                // standard gravitational parameter
	double invdist;															// inverse distance between particle and central planet
	
	if (id < numParticles) {
		mu         = m[0] + m[id];	
		invdist    = rnorm3d(r[id]-r[0], r[numParticles+id]-r[numParticles], r[2*numParticles+id]-r[2*numParticles]);		
	
		L[0]  	   = (r[numParticles+id]-r[numParticles])*v[2*numParticles+id] - (r[2*numParticles+id]-r[2*numParticles])*v[numParticles+id];
		L[1]  	   = (r[2*numParticles+id]-r[2*numParticles])*v[numParticles+id] - (r[numParticles+id]-r[0])*v[2*numParticles+id];
		L[2]  	   = (r[id]-r[0])*v[numParticles+id]   - (r[numParticles+id]-r[numParticles])*v[id];

		eccTemp[0] = (1./mu) * (v[numParticles+id]*L[2] - v[2*numParticles+id]*L[1]) - (r[id]-r[0])   * invdist;
		eccTemp[1] = (1./mu) * (v[2*numParticles+id]*L[0] - v[id]*L[2])   - (r[numParticles+id]-r[numParticles]) * invdist;
		eccTemp[2] = (1./mu) * (v[id]*L[1]   - v[numParticles+id]*L[0]) - (r[2*numParticles+id]-r[2*numParticles]) * invdist;

		ecc[id]    = norm3d(eccTemp[0], eccTemp[1], eccTemp[2]); // real eccentricity
	}
}

/// @brief Reduce last warp (unrolled) in reduction for the A2 operator and B operator. For more details,
///		   visit https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__device__ void warpReduce(volatile double* sdata, int tid) {
	// All statements evaluated at compile time
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

/// @brief Reduction kernel for A2 operator for particle 0 and B operator for particle 1. For more details,
///  	   visit https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf 
template <unsigned int blockSize>
__global__ void reduce(double *g_idata, double *g_odata, unsigned int n) {
    extern __shared__ double sdata[];
	unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
     	sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/// @brief Used to calculate the total linear momentum of the system

/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable numParticles: The number of particles in the simulation: n
/// @variable P:			Holds the total momentum
void linMomentum(double* v, double* m, int numParticles, double *P) {
    *P = 0; 		 // total linear momentum
    double  plin[3]; // adds up total Px, Py, Pz components

    for (int i = 0; i < numParticles; i++) {
        plin[0] += m[i]*v[i], plin[1] += m[i]*v[numParticles+i], plin[2] += m[i]*v[2*numParticles+i];
    *P = sqrt(pow(plin[0], 2) + pow(plin[1], 2) + pow(plin[2], 2));
    }
}

/// @brief Used to calculate the total mass of the system

/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable numParticles: The number of particles in the simulation: n
/// @variable M:			Total mass of the system
void totalMass(double *m, int numParticles, double* M) {
    *M = 0;
    for (int i = 0; i < numParticles; i++)
        *M += m[i];
}

/// @brief Used to calculate the total angular momentum of the system

/// @variable r:            A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable numParticles: The number of particles in the simulation: n
/// @variable L:			Total angular momenum of the system
void angMomentum(double* r, double* v, double* m, int numParticles, double *L) {
	*L = 0;
    double Ltemp[3];
	double crossP[3]; // store cross product result
    double dirvec[3]; // distance from planet
    double	p[3]; 	  // linear momentum

    for (int i = 1; i < numParticles; i++) {
        dirvec[0] = -r[0]+r[i], dirvec[1] = -r[numParticles]+r[numParticles+i], dirvec[2] = -r[2*numParticles]+r[2*numParticles+i];
             p[0] = m[i]*v[i],      p[1] = m[i]*v[numParticles+i],	 p[2] = m[i]*v[2*numParticles+i];
        crossProduct2(dirvec, p, crossP);
    	Ltemp[0] += crossP[0], Ltemp[1] += crossP[1], Ltemp[2] += crossP[2];
	}
	*L = sqrt(pow(Ltemp[0], 2) + pow(Ltemp[1], 2) + pow(Ltemp[2], 2));
}

/// @brief 					Calculates the total energy of the system: kinetic energy plus potential energy

/// @variable r:            A 1D array of size 3n of the particle positions indexed as [x0,x1,...,x(n-1),y0,y1,...,y(n-1),z0,z1,...,z(n-1)]
/// @variable v:            A 1D array of size 3n of the particle velocities indexed as [v0x,v1x,...,v(n-1)x,v0y,v1y,...,v(n-1)y,v0z,v1z,...,v(n-1)z]
/// @variable m:            A 1D array of size n of the particle masses indexed as [m0,m1,...,mn]
/// @variable numParticles: The number of particles in the simulation: n
/// @variable eps:			The gravitational softening parameter
double energynew(double* r, double* v, double* m, int numParticles, double eps) {
    double T = 0;  // kinetic energy
    double U = 0;  // potential energy
    double invdist;

    // to hold the vector that points between particle i and particle j
    double* dirvec = (double*)malloc(3 * sizeof(double));

    for (int i = 0; i < numParticles; i++) {
     	T += 0.5 * m[i] * (pow(v[i], 2) + pow(v[numParticles+i], 2) + pow(v[2*numParticles+i], 2));

        if (i > 0) {
			dirvec[0] = -r[0]+r[i], dirvec[1] = -r[numParticles]+r[numParticles+i], dirvec[2] = -r[2*numParticles]+r[2*numParticles+i];

            invdist = m[i] / sqrt(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2));
            U -= m[0] * invdist;
        }
		if (i > 1) {
			dirvec[0] = -r[0]+r[i], dirvec[1] = -r[numParticles]+r[numParticles+i], dirvec[2] = -r[2*numParticles]+r[2*numParticles+i];

            invdist = m[i] / sqrt(pow(dirvec[0], 2) + pow(dirvec[1], 2) + pow(dirvec[2], 2) + eps*eps);
            U -= m[1] * invdist;
        }
    }
    free(dirvec);

    return T + U;
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// Perform the simulation
extern "C" {
void runSim2(double *r_h, double *v_h, double *m_h, double dt, int numParticles, int n, double eps, int numSteps, double *ecc_h, double *status_h, double *rSatellites_h) {
	// Declare useful variables
    size_t i, j; 
	const unsigned int warpSize = 32;
	size_t N                    = 3 * numParticles;   
    size_t N_bytes              = N * sizeof(double);
	double rH 					= 5.37e10/8.8605e9; // scaled 
	double L;	double P;   double M;   double K;   // final angular momentum, linear momentum, mass and energy
	double L0;  double P0;  double M0;  double K0;  // initial angular momentum, linear momentum, mass and energy
	double semMjrAxis;

	/*cudaError_t stat1 = cudaMallocHost((void**)&r_h, N_bytes);
	if (stat1 != cudaSuccess)
  		printf("Error allocating pinned host memory for position\n");

    cudaError_t stat2 = cudaMallocHost((void**)&v_h, N_bytes);
    if (stat2 != cudaSuccess)
        printf("Error allocating pinned host memory for velocity\n");

    cudaError_t stat3 = cudaMallocHost((void**)&m_h, N_bytes/3);
    if (stat3 != cudaSuccess)
        printf("Error allocating pinned host memory for mass\n");*/

	// Make sure the number of particles is multiple of twice the warp size (2*32)
	// for efficiency and reduction
    if (numParticles % (2*warpSize) != 0) {
    	printf("Error: The number of particles must be a multiple of two times the warp size (2*32).\n");
        return;
    }

	// Allocate arrays on device
    double *r_d, *v_d, *m_d, *ecc_d, *varr_d, *rSatellites_d, *status_d, *vTemp_d;
	cudaMalloc((void**) &r_d, N_bytes);
    cudaMalloc((void**) &v_d, N_bytes);
    cudaMalloc((void**) &m_d, N_bytes/3);
	cudaMalloc((void**) &varr_d, N_bytes);
	cudaMalloc((void**) &status_d, N_bytes/3);
	cudaMalloc((void**) &ecc_d, N_bytes/3);
	cudaMalloc((void**) &rSatellites_d, 2*sizeof(double));
	cudaMalloc((void**) &vTemp_d, numParticles/512*sizeof(double));

	// Copy arrays from host to device
    cudaMemcpy(r_d, r_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h, N_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_d, m_h, N_bytes/3, cudaMemcpyHostToDevice);
	cudaMemcpy(status_d, status_h, N_bytes/3, cudaMemcpyHostToDevice);
	cudaMemcpy(rSatellites_d, rSatellites_h, 2*sizeof(double), cudaMemcpyHostToDevice);

	// calculate initial quantities
    /*angMomentum(r_h, v_h, m_h, numParticles, &L0);
    linMomentum(v_h, m_h, numParticles, &P0);
    totalMass(m_h, numParticles, &M0);
    K0 = energynew(r_h, v_h, m_h, numParticles, eps);*/

	for (i = 0; i < numSteps; i++) {
        // One time step
    	for (j = 0; j < n; j++) {
        	//collision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);  								// check for collision (must do before A1)
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);													// if any status got set to a different value, conserve momentum
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);														// change updated status value to 0 and remove it from the simulation
        	A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));																				// update positions
        	//mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);														// check for merger
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);													// conserve momentum
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);														// change updated status value to 0 and	remove it from the simulation

        	A2_kernel<<<numParticles/64, 64>>>(r_d, v_d, m_d, dt/(2*n), varr_d, status_d, numParticles); 								// update velocities due to particle 0
        	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[0], numParticles);								// reduction for x-component of v0
        	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[numParticles], numParticles); 	// reduction for y-component of v0
        	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2*numParticles], numParticles); // reduction for z-component of v0

        	//collision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);									// same as above
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
        	A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
        	//mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
    	}
    	B_kernel<<<numParticles/64, 64>>>(r_d, v_d, m_d, varr_d, dt, numParticles, status_d, eps);										// update velocities of other particles due to embryo
    	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[1], numParticles);									// reduction for x-component of v1
    	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[numParticles+1], numParticles);   	// reduction for y-component of v0
    	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2*numParticles+1], numParticles);	// reduction for z-component of v0

    	for (j = 0; j < n; j++) {
        	//collision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
        	A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
        	//mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);

        	A2_kernel<<<numParticles/64, 64>>>(r_d, v_d, m_d, dt/(2*n), varr_d, status_d, numParticles);
        	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d, &v_d[0], numParticles);
        	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+numParticles, &v_d[numParticles], numParticles);
        	reduce<512><<<1, numParticles/2, numParticles*sizeof(double)>>>(varr_d+2*numParticles, &v_d[2*numParticles], numParticles);

        	//collision<<<numParticles/64, 64>>>(r_d, v_d, status_d, rSatellites_d, numParticles, dt);
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
        	A1_kernel<<<N/512, 512>>>(r_d, v_d, dt/(4*n));
        	//mergeEject<<<numParticles/64, 64>>>(r_d, status_d, numParticles, rH);
        	//consMomentum<<<1, 1>>>(v_d, m_d, status_d, numParticles, rSatellites_d);
        	//statusUpdate<<<N/64, 64>>>(r_d, v_d, m_d, status_d, numParticles);
    	}

		// after each time-step, copy arrays from device back to host to calculate quantities if you wish
    	/*cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
    	cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
    	cudaMemcpy(m_h, m_d, N_bytes/3, cudaMemcpyDeviceToHost);
	    cudaMemcpy(status_h, status_d, N_bytes/3, cudaMemcpyDeviceToHost);
    	cudaMemcpy(rSatellites_h, rSatellites_d, 2*sizeof(double), cudaMemcpyDeviceToHost);

		// would be easier to make these device functions so only 4 numbers need to be copied back each time-step
    	angMomentum(r_h, v_h, m_h, numParticles, &L);
    	linMomentum(v_h, m_h, numParticles, &P);
    	totalMass(m_h, numParticles, &M);
		K = energynew(r_h, v_h, m_h, numParticles, eps);
		calcEccentricity<<<numParticles/64, 64>>>(r_d, v_d, m_d, ecc_d, numParticles);
		cudaMemcpy(ecc_h, ecc_d, N_bytes/3, cudaMemcpyDeviceToHost);
		semMjrAxis = (m_h[0]+m_h[1])*sqrt(r_h[0]*r_h[0]+r_h[1]*r_h[1]+r_h[2]*r_h[2])/(2*(m_h[0]+m_h[1])-sqrt((r_h[0]-r_h[3])*(r_h[0]-r_h[3])+(r_h[1]-r_h[4])*(r_h[1]-r_h[4])+\
			(r_h[2]-r_h[5])*(r_h[2]-r_h[5]))*sqrt(v_h[3]*v_h[3]+v_h[4]*v_h[4]+v_h[5]*v_h[5])*sqrt(v_h[3]*v_h[3]+v_h[4]*v_h[4]+v_h[5]*v_h[5]));

		printf("%.15lf %.15lf %.15lf %.15lf %.15lf %.15lf\n", abs((L-L0)/L0), abs((P-P0)/P0), abs((M-M0)/M0), abs((K-K0)/K0), ecc_h[1], semMjrAxis);*/
	}

	// Copy arrays from device back to host
    cudaMemcpy(r_h, r_d, N_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_h, v_d, N_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(m_h, m_d, N_bytes/3, cudaMemcpyDeviceToHost);
	cudaMemcpy(status_h, status_d, N_bytes/3, cudaMemcpyDeviceToHost);
	cudaMemcpy(rSatellites_h, rSatellites_d, 2*sizeof(double), cudaMemcpyDeviceToHost);

	// Print relevant information
	int h = 0;
	printf("Embryo radius = %.16lf\n", rSatellites_h[0]);
	for (int kk = 0; kk < numParticles; kk++) {
    	if (status_h[kk] == 0) {
        	printf("Index: %d\n", kk);
            printf("New Position\n");
            printf("%.16lf %.16lf %.16lf\n", r_h[kk], r_h[numParticles+kk], r_h[2*numParticles+kk]);
            printf("New Velocity\n");
            printf("%.16lf %.16lf %.16lf\n", v_h[kk], v_h[numParticles+kk], v_h[2*numParticles+kk]);
        	h += 1;
		}
    }
	printf("%d\n", h);
	printf("New Mass Planet\n");
	printf("%.16lf\n", m_h[0]);
    printf("New Velocity Planet\n");
    printf("%.16lf %.16lf %.16lf\n", v_h[0], v_h[numParticles], v_h[2*numParticles]);
	printf("New Mass Embryo\n");
	printf("%.16lf\n", m_h[1]);
   	printf("New Velocity Embryo\n");
    printf("%.16lf %.16lf %.16lf\n", v_h[1], v_h[numParticles+1], v_h[2*numParticles+1]);
	printf("After %d time step(s):\n", numSteps);
    printf("r\n");
    for (i = 0; i < 3; i ++)
	    printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[numParticles+i], r_h[2*numParticles+i]);
    printf("...\n");
    for (i = numParticles - 3; i < numParticles; i++)
     	printf("%.16lf %.16lf %.16lf\n", r_h[i], r_h[numParticles+i], r_h[2*numParticles+i]);
    printf("\n");
    printf("v\n");
    for (i = 0; i < 3; i ++)
	    printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[numParticles+i], v_h[2*numParticles+i]);
    printf("\n");
    printf("...\n");

    for (i = numParticles - 3; i < numParticles; i ++)
     	printf("%.16lf %.16lf %.16lf\n", v_h[i], v_h[numParticles+i], v_h[2*numParticles+i]);

	// Free allocated memory on host and device
    cudaFree(r_d);
    cudaFree(v_d);
    cudaFree(m_d);
	cudaFree(varr_d);
	cudaFree(status_d);
    cudaFree(ecc_d);
	cudaFree(rSatellites_d);
}
}

