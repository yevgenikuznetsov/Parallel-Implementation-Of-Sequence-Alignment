#ifndef mpiInfo_H_
#define mpiInfo_H_

#define MASTER 0
#define TERMINATION_TAG 1
#define CONTINUE_TAG 0

void sendMissionToSlaves(Info info, MPI_Datatype infoType, MPI_Datatype infoScoreMPIType, FILE *inputFile, int numOfProcesse, MPI_Status status);
void salveCalculation(MPI_Status status, MPI_Datatype infoType, MPI_Datatype infoScoreMPIType);
void readFileAndCheck(Info info, int size, MPI_Datatype infoType, MPI_Datatype infoScoreMPIType, MPI_Status status);
void createMpiInfoStruct(MPI_Datatype *infoMpiType, Info info);
void createMpiInfoBestAlignmentScoreStruct(MPI_Datatype *infoMpiType, InfoBestAlignmentScore scoreInfo);
void printResultToFile(int numOfProcessor, InfoBestAlignmentScore scoreInfo, MPI_Datatype infoScoreType, MPI_Status status);
void sendMissionToSlaves(Info info, MPI_Datatype infoType, FILE *fp);

#endif
