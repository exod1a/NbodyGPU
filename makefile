# makefile

# compile OpenCL files with gcc FILENAME.c -framework OpenCL -o FILENAME
# run with ./FILENAME  Don't have to link .cl files

# compiler and flags
CXX = gcc
CXXCUDA = nvcc
FLAGS = -O3 -Xcompiler -fPIC -shared

all: runSim2.so #runSim.so

#runSim.so: runSim.cu
#	${CXXCUDA} ${FLAGS} -lineinfo -o runSim.so runSim.cu

runSim2.so: runSim2.cu
	${CXXCUDA} ${FLAGS} -lineinfo -o runSim2.so runSim2.cu

# delete .so and .pyc files
clean:
	rm -f *.so *.pyc a.out
