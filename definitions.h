/*
 * mp.h
 *
 *  Created on: Jul 20, 2020
 *      Author: linuxu
 */
#ifndef MP_H_
#define MP_H_


#define WEIGHT_NUM 4
#define DNA_MAX_SIZE 3000
#define RNA_MAX_SIZE 2000
#define HYPHEN 1.3

typedef struct{
	float weightNumber[4];
	char dna[3000];
	int rnaNumber;
	char rna[2000];
	int line;
}Info;

typedef struct{
	int firstCalculate;
	float score;
	int n;
	int k;
	int lineNumber;
}InfoBestAlignmentScore;

InfoBestAlignmentScore compareBetweenDNAtoRNA(InfoBestAlignmentScore scoreInfo, Info info);
InfoBestAlignmentScore trakingOffset(InfoBestAlignmentScore scoreInfo, Info infoint);
int isItMaximal(float result, float scoreInfoScore);
InfoBestAlignmentScore calculateResult(float *result, int hyphen, int offset, int size, InfoBestAlignmentScore scoreInfo);
InfoBestAlignmentScore compareBetweenScore(float newScore, int hyphen, int offset, InfoBestAlignmentScore scoreInfo);
float calculateTotalResult(float *result, int size);
void freeAllocation(float *numberArray);


int computeOnGPU(Info info, int offset, float *result);


#endif /* MP_H_ */
