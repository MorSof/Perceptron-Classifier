#pragma once

#define INPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\Perceptron-Classifier-master\\Perceptron Classifier\\Perceptron Classifier\\Input.txt"
#define OUTPUT_FILE_NAME "C:\\Users\\cudauser\\Desktop\\Perceptron-Classifier-master\\Perceptron Classifier\\Perceptron Classifier\\Output.txt"
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
	int* group;

}Points;

FILE* readParamsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId);
void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC);
void readPointsFromFile(const char* fileName, FILE* fp, Points* points, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId);
void broadcastPoints(Points* points, int N, int K);
void masterHandleSlaves(int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, double start);
void masterAlone(Points points, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, double start, int myId);
void slavesWork(Points points, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, int myId);
int binaryClassificationAlgorithm(int N, int K, Points* points, double* weights, double a, int LIMIT, double t, double proc_dt, double* q, double QC, int* results, Points* dev_Points, double* dev_weights, int* dev_results, int myId);
void fixWeights(int K, int sign, double* coords, double* weights, double a);
int calculateWeightFuncOMP(double* dev_coords, double* dev_weights, int K, int id);
double checkQualityOfClassifier(int Nmiss, int N);
int searchForFirstSuccess(double* qArray, int numOfWorkingProccesses);
void sendMessageToAllSlaves(int numOfWorkingSlaves, int message, int tag);
void dismissUnemployedProcesses(int myCurrentJobIndex, int numOfJobs, int myId);
void initWeights(double* weights, int K);
void writeResultToFile(const char* fileName, double* weights, int K, double t, double q, int result);
void allocateCpuMemory(int N, int K, Points* points, double** weights);
void freeCpuMemory(Points* points, double* weights);


