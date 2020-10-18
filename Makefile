build:
	mpicxx -fopenmp -c main.c -o main.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP main.o cudaFunctions.o  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 5 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 7 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP
