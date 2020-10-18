#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "definitions.h"
#include <stdio.h>
#include <string.h>

#define NUMOFCONSERVATIVEGROUPS 9
#define NUMOFSEMICONSERVATIVEGROUPS 12
#define NNMOFWIEGHT 4
#define STAR 0
#define COLON 1
#define SIGN 2
#define SPACE 3

__global__ void compareBetweenDNAandRNA(char *DNA, char *RNA, float *result, int size, float *weight);
__device__ void compareTwoChar(float *result, char dnaChar, char rnaChar, int i, float *weight);
__device__ int isBelongToConservativeGroups(char first, char second);
__device__ int isBelongToSemiConservativeGroups(char first, char second);
__device__ int isTheCharsInTheSameGroupAndSameString(const char *str, char first, char second);
__device__ int isTheCharInTheString(const char *str, char ch);

void errorChecking(cudaError_t err);
void freeCharacterStringAllocation(cudaError_t err, char *characterString);
void freeNumberAllocation(cudaError_t err, float *numbers);

/* Compare between the char in DNA to char in RNA */
__global__ void compareBetweenDNAandRNA(char *DNA, char *RNA, float *result, int size, float *weight) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size){
		compareTwoChar(result, DNA[i], RNA[i], i, weight);
	}  
}

/* The function check for each two char what their weight is */
__device__ void compareTwoChar(float *result, char dnaChar, char rnaChar, int i, float *weight){

	result[i] = -(weight[SPACE]);

	if(dnaChar == rnaChar){	
		result[i] = weight[STAR];
	}
	else if (isBelongToConservativeGroups(dnaChar, rnaChar) > 0){
		result[i] = -(weight[COLON]);
	}
	else if (isBelongToSemiConservativeGroups(dnaChar, rnaChar) > 0){
		result[i] = -(weight[SIGN]);
	}
	else{
		result[i] = -(weight[SPACE]);
	}
}

/* The function check if two char belong to conservative groups */
__device__ int isBelongToConservativeGroups(char first, char second){
	
	const char *conservativeGroups[NUMOFCONSERVATIVEGROUPS]= {
		"NDEQ","NEQK","STA","MILV","QHRK","NHQK","FYW","HY","MILF"
	};
	
	// Go over each string in this group and check if two char exist in stirng
	for(int j = 0 ; j < NUMOFCONSERVATIVEGROUPS ; j++){
		if (isTheCharsInTheSameGroupAndSameString(conservativeGroups[j], first, second) != 0) {
				return 1; 
		}	
	}
		
	return 0;
}

/* The function check if two char belong to semi conservative groups */
__device__ int isBelongToSemiConservativeGroups(char first, char second){

	const char *semiConservativeGroups[NUMOFSEMICONSERVATIVEGROUPS]= {
		"SAG","ATV","CSA","SGND","STPA","STNK","NEQHRK","NDEQHK","SNDEQK"," ","HFY","FVLIM"	
	};
	
	// Go over each string in this group and check if two char exist in stirng
	for(int j = 0 ; j < NUMOFSEMICONSERVATIVEGROUPS ; j++){
		if (isTheCharsInTheSameGroupAndSameString(semiConservativeGroups[j], first, second) != 0) {
				return 1; 
		}
	}
	
	return 0;
}

/* The function check if two char exist in the same stirng */
__device__ int isTheCharsInTheSameGroupAndSameString(const char *str, char first, char second){

	int resultFromFirstChar = isTheCharInTheString(str,first);
	int resultFromSecondChar = isTheCharInTheString(str,second);

	if( (resultFromFirstChar != 0) & (resultFromSecondChar != 0)){
		return 1; 
	}

	return 0; 
}

/* The function check if the char exist in string */
__device__ int isTheCharInTheString(const char *str, char ch){

	int i = 0 ;
	
	while(str[i] != '\0'){

		if(str[i] == ch){
			return 1;	
		}
		
		i++;
	}

	return 0;
}

int computeOnGPU(Info info, int offset, float *result){
    
	cudaError_t err = cudaSuccess;
	
	int sizeOfRna = strlen(info.rna) - 1;
	size_t size = sizeOfRna * sizeof(char);
	size_t sizeForResult = sizeOfRna * sizeof(float);

	float weight[NNMOFWIEGHT] = {info.weightNumber[STAR], info.weightNumber[COLON], info.weightNumber[SIGN],
					info.weightNumber[SPACE]};
  
	// Allocate memory on GPU to copy the data from the host
	char *d_RNA;
	err = cudaMalloc((void **)&d_RNA, size);
	errorChecking(err);

	char *d_DNA;
	err = cudaMalloc((void **)&d_DNA, size);
	errorChecking(err);

	float *d_Weight;
	err = cudaMalloc((void **)&d_Weight, NNMOFWIEGHT*sizeof(float));
	errorChecking(err);

	float *d_C;
	err = cudaMalloc((void **)&d_C, sizeForResult);
	errorChecking(err);

	// Copy data from host to the GPU memory
	err = cudaMemcpy(d_RNA, info.rna, size, cudaMemcpyHostToDevice);
 	errorChecking(err);
	
	err = cudaMemcpy(d_DNA, &(info.dna[offset]), size, cudaMemcpyHostToDevice);
	errorChecking(err);

	err = cudaMemcpy(d_Weight, weight, NNMOFWIEGHT*sizeof(float), cudaMemcpyHostToDevice);
	errorChecking(err);
    
	// Launch the Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =(sizeOfRna + threadsPerBlock - 1) / threadsPerBlock;
	compareBetweenDNAandRNA<<<blocksPerGrid, threadsPerBlock>>>(d_DNA, d_RNA, d_C, sizeOfRna, d_Weight);
	err = cudaGetLastError();
	errorChecking(err);
	
	// Copy the result from GPU to the host memory
	err = cudaMemcpy(result, d_C, sizeForResult, cudaMemcpyDeviceToHost);
	errorChecking(err);

	// Free allocated memory on GPU
	freeCharacterStringAllocation(err, d_DNA);
	freeCharacterStringAllocation(err, d_RNA);
	freeNumberAllocation(err, d_C);
	freeNumberAllocation(err, d_Weight);
    
  return 0;
}

void errorChecking(cudaError_t err){

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void freeCharacterStringAllocation(cudaError_t err, char *characterString){

	if (cudaFree(characterString) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
      	        exit(EXIT_FAILURE);
	}
}

void freeNumberAllocation(cudaError_t err, float *numbers){

	if (cudaFree(numbers) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
      	        exit(EXIT_FAILURE);
	}
}

