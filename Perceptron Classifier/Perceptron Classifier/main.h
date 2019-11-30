#pragma once

#define INPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\Prceptron - Cuda\\Prceptron - Cuda\\Input.txt"
#define OUTPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\Prceptron - Cuda\\Prceptron - Cuda\\Output.txt"
#define MASTER 0
#define WEIGHTS_TAG 2
#define GLOBAL_SUCCESS_TAG 3
#define GLOBAL_FAIL_TAG 4
#define CONTINUATION_TAG 5
#define RESULT_TAG 6
#define FAIL -1
#define SUCCESS 1

typedef struct {

	double* coordinantes;
	double* velocity;
	int group;

}Point;

typedef struct {

	double* coordinantes;
	double* velocity;
	int* group;

}Points;

void masterHandleSlaves(int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, clock_t c);
void masterAlone(Points points, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, clock_t c, int myId);
void slavesWork(Points points, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, int myId);
FILE* readParamsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId);
void readPointsFromFile(const char* fileName, FILE* fp, Points* points, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId);
int searchForFirstSuccess(double* qArray, int numOfWorkingProccesses);
int binaryClassificationAlgorithm(int N, int K, Points* points, double* weights, double a, int LIMIT, double t, double proc_dt, double* q, double QC, int* results, Points* dev_Points, double* dev_weights, int* dev_results, int myId);
void fixWeights(int K, int sign, double* coords, double* weights, double a);
int calculateWeightFuncOMP(double* dev_coords, double* dev_weights, int K, int id);
double checkQualityOfClassifier(int Nmiss, int N);
void initWeights(double* weights, int K);
void printWeights(double* weights, int K, int myId);
void writeResultToFile(const char* fileName, double* weights, int K, double t, double q, int result);
void allocateCpuMemory(int N, int K, Points* points, double** weights);
void freeCpuMemory(Points* points, double* weights);
void dismissUnemployedprocesses(int myCurrentJobIndex, int numOfJobs, int myId);
void sendMessageToAllSlaves(int numOfWorkingSlaves, int message, int tag);
void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC);
void broadcastPoints(Points* points, int N, int K);


//------------------------------------------------------------
void writePointsToFileTest(const char* fileName, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, Point* pointsArr);
void writeOnePointToFile(FILE* fp, Point* pPoint, int K);
Point createOnePoint(int K);
Point* createPointArr(int N, int K);
void printOnePoint(Points* points, int i, int K);
void printPointArr(Points* points, int N, int K, int myId);
void printQarray(double* qArray, int numOfWorkingSlaves);
