/*
 ============================================================================
 Name        : main.c
 Author      : yevgeni kuznetsov
 Version     :
 Copyright   : 
 Description : MPI + CUDA + OPENMP
 ============================================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include "mpi.h"
#include "definitions.h"
#include "mpiInfo.h"

int main(int argc, char* argv[]){
	int  rank;
	int  size;

	Info info;
	InfoBestAlignmentScore scoreInfo;
	
	// MPI init variables
	MPI_Datatype infoMPIType;
	MPI_Datatype infoScoreMPIType;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	MPI_Status status;
	
	// Defining structures for MPI
	createMpiInfoStruct(&infoMPIType, info);
	createMpiInfoBestAlignmentScoreStruct(&infoScoreMPIType, scoreInfo);
	
	// Handle by MASTER
	if (rank == 0)
	{	
		readFileAndCheck(info, size, infoMPIType, infoScoreMPIType, status);
	}
	// Handle by SLAVE
	else
	{
		salveCalculation(status, infoMPIType,infoScoreMPIType);
	}
	
	MPI_Finalize(); 
		
	return 0;
}

/* The function equates DNA to RNA (goes over all offsets).
   Calculate the best result (including calculating mutations) */
InfoBestAlignmentScore trakingOffset(InfoBestAlignmentScore scoreInfo, Info info){
	
	int offset, lengh, j, i;
	float *firsrArrayResult, *secondArrayResult;
	
	lengh = strlen(info.rna)-1;

	firsrArrayResult = (float*)malloc(sizeof(float) * lengh);
	secondArrayResult = (float*)malloc(sizeof(float) * lengh);
	

	for (offset = 0; offset < (strlen(info.dna) - strlen(info.rna)); offset++) {
		
		// Calculate first result array with CUDA
		if(computeOnGPU(info, offset, firsrArrayResult) != 0){
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
		
		// Calculate second result array with CUDA
		if(computeOnGPU(info, offset + 1, secondArrayResult) != 0){
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
		
		scoreInfo = calculateResult(firsrArrayResult, 0, offset, lengh ,scoreInfo); 
		
		float leftSum = firsrArrayResult[0];
		float rightSum = 0; 
		
		// Calculte score with the first hyphen
		for (j = 1; j < lengh; j++) {
			rightSum += secondArrayResult[j];
		}

		scoreInfo = compareBetweenScore(leftSum + rightSum - HYPHEN, 1, offset, scoreInfo);
		
		// Calculate score for each hyphen(excluding the first hyphen)
		for (i = 2; i < strlen(info.rna); i++) {
			leftSum += firsrArrayResult[i-1];
			rightSum -= secondArrayResult[i-1];
			
			scoreInfo = compareBetweenScore(leftSum + rightSum - HYPHEN, i, offset, scoreInfo);
		}
	}

	scoreInfo = calculateResult(secondArrayResult, 0, offset, lengh ,scoreInfo); 

	freeAllocation(firsrArrayResult); // Free allocation	
	freeAllocation(secondArrayResult); // Free allocation
 
	return scoreInfo;
}

/* Compare between score */
InfoBestAlignmentScore compareBetweenScore(float newScore, int hyphen, int offset, InfoBestAlignmentScore scoreInfo){

	InfoBestAlignmentScore infoscore;
	
	// Check if new score is maximum
	if(isItMaximal(newScore, scoreInfo.score) == 1){
		infoscore.n = offset;
		infoscore.score = newScore;
		infoscore.k = hyphen;
		infoscore.lineNumber = scoreInfo.lineNumber;
	}else {
		infoscore = scoreInfo;
	}

	return infoscore;
}

/* Calculate the total result for RNA and check if the result is maximum */
InfoBestAlignmentScore calculateResult(float *result, int hyphen, int offset, int size, InfoBestAlignmentScore scoreInfo){
	float score = 0;
	InfoBestAlignmentScore infoscore;
	
	score = calculateTotalResult(result, size);
	
	// Check if we calculate this score first time for this RNA
	if(scoreInfo.firstCalculate == 1){
		infoscore.k = 0;
		infoscore.n = 0;
		infoscore.score = score;
		infoscore.lineNumber = scoreInfo.lineNumber;
		
		scoreInfo.firstCalculate = 0;
	}
	else {
		infoscore = compareBetweenScore(score, hyphen, offset, scoreInfo); // Compare between score
	}

	return infoscore;
}

/* Calculate the total result with OpenMP */
float calculateTotalResult(float *result, int size){
	float score = 0;
	int i;
	
	omp_set_num_threads(8); // Set the num of threds in OpenMP

// Use OpenMP to sum the total result
#pragma omp parallel for reduction (+:score) private(i)
	for(i = 0 ; i < size; i++){
		score = score + result[i];
	}
	
	return score;
}

/* The function check if the result is maximum */
int isItMaximal(float result, float scoreInfoScore){

	if(scoreInfoScore < result){
		return 1;
	}

	return 0;
}

/* Function will create a new MPI_Datatype */
void createMpiInfoStruct(MPI_Datatype *infoMpiType, Info info){

	int blockLenghts[5] = {4, 3000, 1, 2000, 1};
	MPI_Aint disp[5];
	MPI_Datatype types[5] = {MPI_FLOAT, MPI_CHAR, MPI_INT, MPI_CHAR, MPI_INT};

	disp[0] = (char*) &info.weightNumber - (char*) &info;
	disp[1] = (char*) &info.dna - (char*) &info;
	disp[2] = (char*) &info.rnaNumber - (char*) &info;
	disp[3] = (char*) &info.rna - (char*) &info;
	disp[4] = (char*) &info.line - (char*) &info;

	MPI_Type_create_struct(5, blockLenghts, disp, types, infoMpiType);
	MPI_Type_commit(infoMpiType);
}

/* Function will create a new MPI_Datatype */
void createMpiInfoBestAlignmentScoreStruct(MPI_Datatype *infoMpiType, InfoBestAlignmentScore scoreInfo){

	int blockLenghts[5] = {1, 1, 1, 1, 1};
	MPI_Aint disp[5];
	MPI_Datatype types[5] = {MPI_INT, MPI_FLOAT, MPI_INT, MPI_INT, MPI_INT};

	disp[0] = (char*) &scoreInfo.firstCalculate - (char*) &scoreInfo;
	disp[1] = (char*) &scoreInfo.score - (char*) &scoreInfo;
	disp[2] = (char*) &scoreInfo.n - (char*) &scoreInfo;
	disp[3] = (char*) &scoreInfo.k - (char*) &scoreInfo;
	disp[4] = (char*) &scoreInfo.lineNumber - (char*) &scoreInfo;

	MPI_Type_create_struct(5, blockLenghts, disp, types, infoMpiType);
	MPI_Type_commit(infoMpiType);
}

/* The function will read the data from the file and then send this data to slave */
void readFileAndCheck(Info info, int size, MPI_Datatype infoType, MPI_Datatype infoScoreMPIType, MPI_Status status){

	FILE *inputFile;
	int i;
	
	// Open the file to read
	inputFile = fopen("/home/linuxu/input.txt","r");
	if(!inputFile){
		printf("ERROR opening file");
		exit(1);
	}
	
	// Save weight number from file
	for(i = 0 ; i < WEIGHT_NUM ; i++){
		if(i != WEIGHT_NUM - 1){
			fscanf(inputFile, "%f", &info.weightNumber[i]);
		}
		else {
			fscanf(inputFile, "%f\n", &info.weightNumber[i]);
		}
	}
	
	// Read dna from the file 
	fgets(info.dna, DNA_MAX_SIZE, inputFile);
	// Read the number of RNAs in the file
	fscanf(inputFile, "%d\n", &info.rnaNumber);
	
	// Send RNA and data to SLAVE
	sendMissionToSlaves(info, infoType, infoScoreMPIType, inputFile, size, status);

}

/* The function print the results in the file.
   The MASTER send work to slave. Each slave recieve RNA and information to calculat best result */
void sendMissionToSlaves(Info info, MPI_Datatype infoType, MPI_Datatype infoScoreMPIType, FILE *inputFile, int numOfProcesses, MPI_Status status){
	
	InfoBestAlignmentScore infoScore;

	FILE *outputFile;	

	int message = 0;
	int numberOfMission = 0;
	int slaveNumber;
	
	int freeSlave = (numOfProcesses - 1 > info.rnaNumber) ? (info.rnaNumber + 1) :numOfProcesses;
	// Open a file for write result
	outputFile = fopen("/home/linuxu/output.txt", "w");
	if(!outputFile){
		printf("ERROR! opening file");
	}
	
	// Send tasks to all SLAVES
	for(slaveNumber = 1 ; slaveNumber < numOfProcesses ; slaveNumber++){
		// Check if we need the slave to perform the task
		if(slaveNumber > info.rnaNumber){
			MPI_Send(&message, 1, MPI_INT, slaveNumber, TERMINATION_TAG, MPI_COMM_WORLD);
		}
		else{
			MPI_Send(&message, 1, MPI_INT, slaveNumber, CONTINUE_TAG, MPI_COMM_WORLD);
			
			info.line = numberOfMission + 1; // Save the number of mission
			fgets(info.rna, RNA_MAX_SIZE, inputFile); // Read RNA from the file
				
			MPI_Send(&info, 1, infoType, slaveNumber, CONTINUE_TAG, MPI_COMM_WORLD); // Send information with RNA to slave
			
			numberOfMission++;
		}
	}
	// Check if we have complete all the RAN seq, If not the master will sent new RNA to available slave 
	while(numberOfMission != info.rnaNumber){
		
		MPI_Recv(&infoScore, 1, infoScoreMPIType, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Get result from slave
		
		fprintf(outputFile, "Seq2 number %d, THE BEST Alignment Score: %f (n: %d , MS(%d)) \n", infoScore.lineNumber, infoScore.score, infoScore.n, infoScore.k); // Print result to the file
		
		MPI_Send(&message, 1, MPI_INT, status.MPI_SOURCE, CONTINUE_TAG, MPI_COMM_WORLD);
		
		info.line = numberOfMission + 1; // Save the number of mission
		fgets(info.rna, RNA_MAX_SIZE, inputFile);// Read RNA from the file
		
		MPI_Send(&info, 1, infoType, status.MPI_SOURCE, CONTINUE_TAG, MPI_COMM_WORLD); // Send information with new RNA to slave
		
		numberOfMission++;
	}
	// Get result from all slave 
	for(slaveNumber = 1 ; slaveNumber < freeSlave ; slaveNumber++){
		
		MPI_Recv(&infoScore, 1, infoScoreMPIType, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Get result from slave
		
		fprintf(outputFile, "Seq2 number %d, THE BEST Alignment Score: %f (n: %d , MS(%d)) \n", infoScore.lineNumber, infoScore.score, infoScore.n, infoScore.k); // Print result to the file
		
		MPI_Send(&message, 1, MPI_INT, status.MPI_SOURCE, TERMINATION_TAG, MPI_COMM_WORLD); // Send to slave that we finish all mission
	}	

	fflush(stdout);
	fclose(outputFile); // Close output file
	fclose(inputFile); // Close input file
}

/* Each slave recieve information and RNA and calculate. 
   In the end, each slave send to master the best result */
void salveCalculation(MPI_Status status, MPI_Datatype infoType, MPI_Datatype infoScoreMPIType){
	
	int message;
	
	Info info;
	InfoBestAlignmentScore infoScore;
	
	while(1){
		MPI_Recv(&message, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // Get message from master
		
		// Finish work
		if(status.MPI_TAG == TERMINATION_TAG){
				break;
		}
		else{
			MPI_Recv(&info, 1, infoType, MASTER, CONTINUE_TAG, MPI_COMM_WORLD, &status); // Get information and RNA from master
			
			infoScore.firstCalculate = 1;
			infoScore.lineNumber = info.line;
			
			// Check if RNA lenght equals to DNA lenght 
			if( strlen(info.dna) != strlen(info.rna)){
				infoScore = trakingOffset(infoScore, info); // Send all information to calculate the best result 
			}
			else{
				infoScore = compareBetweenDNAtoRNA(infoScore, info); // Send all information to calculate the best result 
			}
					
			MPI_Send(&infoScore, 1, infoScoreMPIType, MASTER, CONTINUE_TAG, MPI_COMM_WORLD); // Send to master best result
		}
	}
}

/* Compare between RNA and DNA with same lenght */
InfoBestAlignmentScore compareBetweenDNAtoRNA(InfoBestAlignmentScore scoreInfo, Info info){
	float *result, score;
	
	result = (float*)malloc(sizeof(float) * strlen(info.rna)-1);
	
		// Calculate result array with CUDA
		if(computeOnGPU(info, 0, result) != 0){
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
	
	score = calculateTotalResult(result, strlen(info.rna)-1);
	
	scoreInfo.score = score;
	scoreInfo.n = 0;
	scoreInfo.k = 0;
	
	freeAllocation(result);
	
	return scoreInfo;
}
/* Function to free allocation */
void freeAllocation(float *numberArray){
	free(numberArray);
}
