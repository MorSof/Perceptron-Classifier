
#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern cudaError_t allocateCudaMemory(Points* points, int N, int K, Points* dev_points, double** dev_weights, int** dev_results);
extern cudaError_t calcCoordsWithCuda(Points* points, int N, int K, double* weights, double a, int LIMIT, double t, double proc_dt, int* results, int* Nmiss, Points* dev_points, double* dev_weights, int* dev_results, int* numOfBlocks, int* numOfThreadsPerBlock, int myId);
extern int checkAllPointsLimitTimesCuda(Points* points, int N, int K, double* dev_coords, int* dev_group, double* dev_weights, double* weights, double a, int LIMIT, int* results, int* dev_results, int numOfBlocks, int numOfThreadsPerBlock, int myId);
extern void freeCudaMemory(Points* dev_points, double* dev_weights, int* dev_results);
extern int checkAllPointsLimitTimesOMP(Points* points, int N, int K, double* weights, double a, int LIMIT, int* results, int myId);

int main(int argc, char *argv[])
{
	int myId, numOfProcess;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
	int numOfSlaves = numOfProcess - 1;

	int N = 500000;
	int K = 2;
	double dt = 1;
	double tmax = 10;
	double a = 0.2;
	int LIMIT = 200;
	double QC = 0.1;
	double t = 0;
	double* weights;
	int numOfJobs, numOfWorkingProccesses, numOfWorkingSlaves;

	Points points;
	FILE* fp;

	double start = MPI_Wtime();

	if (myId == MASTER)
	{	//Only the Master reads parameters from file
		fp = readParamsFromFile(INPUT_FILE_NAME, &N, &K, &dt, &tmax, &a, &LIMIT, &QC, myId);
	}

	broadcastParameters(&N, &K, &dt, &tmax, &a, &LIMIT, &QC);//broadcast parameters to childeren

	allocateCpuMemory(N, K, &points, &weights); // allocate all cpu's memory variables
	if (myId == MASTER)
	{	//Only the Master reads points from file
		readPointsFromFile(INPUT_FILE_NAME, fp, &points, &N, &K, &dt, &tmax, &a, &LIMIT, &QC, myId);
	}
	numOfJobs = (int)(tmax / dt) + 1;
	numOfWorkingProccesses = numOfProcess > numOfJobs ? numOfJobs : numOfProcess; //if there are more slaves then jobs => the unimployed slaves are not relevant
	numOfWorkingSlaves = numOfSlaves > numOfJobs ? numOfJobs : numOfSlaves; // actual number of slaves that will work (in case there is more slaves then jobs)

	broadcastPoints(&points, N, K); //broadcast points to all slaves

	if (myId == MASTER && numOfProcess > 1) // Slavse doing the algorithem, master will only be the manager
	{
		masterHandleSlaves(N, K, dt, tmax, a, LIMIT, QC, t, weights, numOfJobs, numOfSlaves, numOfWorkingProccesses, numOfWorkingSlaves, start);
	}
	else if (myId == MASTER && numOfProcess == 1) //if there are no slaves, only the master perform the algorithem
	{
		masterAlone(points, N, K, dt, tmax, a, LIMIT, QC, t, weights, numOfJobs, numOfSlaves, numOfWorkingProccesses, numOfWorkingSlaves, start, myId);
	}
	else //Slavse doing the algorithem
	{
		slavesWork(points, N, K, dt, tmax, a, LIMIT, QC, t, weights, numOfJobs, numOfSlaves, numOfWorkingProccesses, numOfWorkingSlaves, myId);
	}

	freeCpuMemory(&points, weights); //free allocations
	MPI_Finalize();
	return EXIT_SUCCESS;
}

void masterHandleSlaves(int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, double start)
{
	MPI_Status status;
	double* qArray = NULL;
	double qLocal;
	int winnerSlavesId, searchResult;
	int round = 0;
	int i;
	qArray = (double*)malloc(numOfWorkingSlaves * sizeof(double)); //if the job will successied it will poccess q as value, else it will poccess FAIL value
	while (numOfJobs > 0)
	{
		numOfWorkingSlaves = numOfSlaves > numOfJobs ? numOfJobs : numOfSlaves; //if there are more slaves then jobs => the unimployed slaves are not relevant			
		for (i = 0; i < numOfWorkingSlaves; i++)
		{
			MPI_Recv(&qLocal, 1, MPI_DOUBLE, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);//recieve local time results from slaves
			qArray[status.MPI_SOURCE - 1] = qLocal;
			numOfJobs--;
		}
		searchResult = searchForFirstSuccess(qArray, numOfWorkingSlaves); //Search for a minimum time success - if it find a success it return its index, else return FAIL
		if (searchResult != FAIL)
		{	// master finds a minimum SUCCESS time
			winnerSlavesId = searchResult + 1; //the id of the slave which provides a success with the minimnum local time
			sendMessageToAllSlaves(numOfWorkingSlaves, /*message*/winnerSlavesId, GLOBAL_SUCCESS_TAG); //send a succession TAG to all slaves with the winner slave as a message - after they will recive the massege they will stop work
			t = t + (searchResult * dt);
			MPI_Recv(weights, K + 1, MPI_DOUBLE, winnerSlavesId, WEIGHTS_TAG, MPI_COMM_WORLD, &status); // recive the weights from the winner slave
			writeResultToFile(OUTPUT_FILE_NAME, weights, K, t, qArray[searchResult], SUCCESS);
			break;
		}
		else if (numOfJobs == 0)
		{	//if there is no more jobs with NO success
			sendMessageToAllSlaves(numOfWorkingSlaves, /*message*/FAIL, GLOBAL_FAIL_TAG); //The master sends failure TAG to all slaves - after they will recive the massege they will stop work
			writeResultToFile(OUTPUT_FILE_NAME, weights, K, t, qArray[numOfWorkingSlaves - 1], FAIL);
		}
		else
		{	//if there is NO succeess but there are availble jobs
			sendMessageToAllSlaves(numOfWorkingSlaves, /*message*/FAIL, CONTINUATION_TAG); // will tell to all slaves to continiue to the next job
			round++;
			t = t + (dt * numOfWorkingSlaves);
		}
	}

	free(qArray);
	double end = MPI_Wtime();
	printf("\nThe program took %lf seconds to execute\n", end - start);

}

void masterAlone(Points points, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, double start, int myId)
{//Master is alone without slaves
	Points dev_points;
	int result;
	double proc_dt;
	double q;
	int round = 0;
	int* results;
	int *dev_results = 0;
	double* dev_weights = 0;

	allocateCudaMemory(&points, N, K, &dev_points, &dev_weights, &dev_results);//Allocate Cuda memory
	results = (int*)malloc(sizeof(int)*N);
	while (numOfJobs > 0)
	{
		numOfJobs--;
		initWeights(weights, K);
		proc_dt = t == 0 ? t : dt; //in the first round it will calculate the intial "t", afterwards it will calculate the advanced "dt";
		result = binaryClassificationAlgorithm(N, K, &points, weights, a, LIMIT, t, proc_dt, &q, QC, results, &dev_points, dev_weights, dev_results, myId); //Starts the binary classification algorithem
		if (result == SUCCESS || numOfJobs == 0)
		{ // a success or no jobs has left
			writeResultToFile(OUTPUT_FILE_NAME, weights, K, t, q, result);
			break;
		}
		else
		{
			// Faild and there are more jobs to be done
			t = t + dt;
			round++;
		}
	}

	freeCudaMemory(&dev_points, dev_weights, dev_results);
	free(results);
	double end = MPI_Wtime();
	printf("\nThe program took %lf seconds to execute\n", end - start);
}

void slavesWork(Points points, int N, int K, double dt, double tmax, double a, int LIMIT, double QC, double t, double* weights, int numOfJobs, int numOfSlaves, int numOfWorkingProccesses, int numOfWorkingSlaves, int myId)
{
	MPI_Status status;
	Points dev_points;
	double qLocal;
	int message;
	int* localResults = NULL;
	int result;
	double proc_dt;
	int myCurrentJobIndex = myId - 1;
	int *dev_results = 0;
	double* dev_weights = 0;

	dismissUnemployedProcesses(myCurrentJobIndex, numOfJobs, myId);//if there are more slaves then jobs, the unimployment slaves will be dismissed
	allocateCudaMemory(&points, N, K, &dev_points, &dev_weights, &dev_results); // Allocate cuda memory
	localResults = (int*)malloc(sizeof(int)*N); // array which include all the local time results
	do
	{
		numOfWorkingSlaves = numOfSlaves > numOfJobs ? numOfJobs : numOfSlaves; //finds out how many slaves will actually work
		proc_dt = myCurrentJobIndex == myId - 1 ? dt * myCurrentJobIndex : dt * numOfWorkingSlaves; //in the first round it will calculate the intial "t", afterwards it will calculate the advanced "dt"
		t = t + proc_dt;
		initWeights(weights, K);//initial the weights to 0
		result = binaryClassificationAlgorithm(N, K, &points, weights, a, LIMIT, t, proc_dt, &qLocal, QC, localResults, &dev_points, dev_weights, dev_results, myId);//Starts the binary classification algorithem
		MPI_Send(&qLocal, 1, MPI_DOUBLE, MASTER, RESULT_TAG, MPI_COMM_WORLD); // send local slave result to master
		MPI_Recv(&message, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // recieve answer from the Master 
		if (status.MPI_TAG == GLOBAL_SUCCESS_TAG || status.MPI_TAG == GLOBAL_FAIL_TAG)
		{	//if the answer is a Global Success or Global Fail, it will not continiue to the next job
			if (status.MPI_TAG == GLOBAL_SUCCESS_TAG && myId == message)
			{ //If this slave is the one who provide the successful weights in minimum time
				MPI_Send(weights, K + 1, MPI_DOUBLE, MASTER, WEIGHTS_TAG, MPI_COMM_WORLD);// will send his successfull weights to master
			}
			break;
		}
		else
		{	//If there is no Global Success or Global Faild we want to continiue to the next job
			myCurrentJobIndex += numOfWorkingSlaves; //Increas his job index by the actuall number of working slaves
			dismissUnemployedProcesses(myCurrentJobIndex, numOfJobs, myId);//if there are more slaves then jobs, the unimployment slaves will be dismissed
		}

	} while (status.MPI_TAG == CONTINUATION_TAG);
	freeCudaMemory(&dev_points, dev_weights, dev_results);
	free(localResults);
}

int binaryClassificationAlgorithm(int N, int K, Points* points, double* weights, double a, int LIMIT, double t, double proc_dt, double* q, double QC, int* localResults, Points* dev_points, double* dev_weights, int* dev_results, int myId)
{
	int Nmiss;
	int numOfBlocks, numOfThreadsPerBlock;
	calcCoordsWithCuda(points, N, K, weights, a, LIMIT, t, proc_dt, localResults, &Nmiss, dev_points, dev_weights, dev_results, &numOfBlocks, &numOfThreadsPerBlock, myId);

	//---------------------Two solutions to the algorithem, please see the README FILE----------------------------------------------

	Nmiss = checkAllPointsLimitTimesCuda(points, N, K, dev_points->coordinantes, dev_points->group, dev_weights, weights, a, LIMIT, localResults, dev_results, numOfBlocks, numOfThreadsPerBlock, myId);
	//Nmiss = checkAllPointsLimitTimesOMP(points, N, K, weights, a, LIMIT, localResults, myId);//Checking all points LIMIT times and return the last iteration Nmiss

	//-----------------------------------------------------------------------------------------------------------------------------------------
	*q = checkQualityOfClassifier(Nmiss, N); //Calculate Ratio (Nmiss/N)
	if (*q < QC)
	{	//Good quality solution
		return SUCCESS;
	}
	//Not a good quality
	*q = (*q) * (-1); //q will be negative
	return FAIL;
}



int calculateWeightFuncOMP(double* coords, double* weights, int K, int id)
{	// Return the sign of a point after place it inside the weights function
	double sum = 0;
	int i;
	const double FIRST_PARAM = 1;
	sum += FIRST_PARAM * weights[0];

	for (i = 0; i < K; i++)
	{
		sum += coords[i + (id * K)] * weights[i + 1];
	}
	return sum < 0 ? -1 : 1;
}

void fixWeights(int K, int sign, double* coords, double* weights, double a)
{	//Calculate the new weights value  W = W + [a*sign(f(P))]P
	int i;
	const double FIRST_PARAM = 1;
	weights[0] = weights[0] + ((a * ((double)(-1 * sign)))*FIRST_PARAM);
	for (i = 0; i < K; i++)
	{
		weights[i + 1] = weights[i + 1] + ((a * ((double)(-1 * sign)))*coords[i]);
	}
}

double checkQualityOfClassifier(int Nmiss, int N)
{	// Return q value
	return ((double)(Nmiss)) / ((double)N);
}


int searchForFirstSuccess(double* qArray, int numOfWorkingProccesses)
{	// Search for the first success among all the working slaves
	int i;
	for (i = 0; i < numOfWorkingProccesses; i++)
	{
		if (qArray[i] >= 0)
		{
			return i;
		}
	}
	return -1;
}

FILE* readParamsFromFile(const char* fileName, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId)
{
	FILE* fp;
	int coordsIndex = 0, velocityIndex = 0;
	fp = fopen(fileName, "r");
	if (!fp)
	{
		printf("Error reading the file! (fp=NULL)\n");
		fflush(NULL);
	}

	fscanf(fp, "%d", N);
	fscanf(fp, "%d", K);
	fscanf(fp, "%lf", dt);
	fscanf(fp, "%lf", tmax);
	fscanf(fp, "%lf", a);
	fscanf(fp, "%d", LIMIT);
	fscanf(fp, "%lf", QC);

	return fp;
}

void readPointsFromFile(const char* fileName, FILE* fp, Points* points, int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC, int myId)
{

	int i, coordsIndex = 0, velocityIndex = 0;

	if (!fp)
	{
		printf("Error reading the file! (fp=NULL)");
	}

	for (i = 0; i < (*N); i++)
	{

		for (coordsIndex; coordsIndex < (*K) * (i + 1); coordsIndex++)
		{
			fscanf(fp, "%lf", &(points->coordinantes[coordsIndex]));
		}

		for (velocityIndex; velocityIndex < (*K) * (i + 1); velocityIndex++)
		{
			fscanf(fp, "%lf", &(points->velocity[velocityIndex]));
		}

		fscanf(fp, "%d", &(points->group[i]));
	}

	fclose(fp);
}

void writeResultToFile(const char* fileName, double* weights, int K, double t, double q, int result)
{

	int i;
	FILE *fp;
	fp = fopen(fileName, "wt");
	if (!fp)
	{
		printf("Cannot Open File '%s'", fileName);
		return;
	}
	if (result == SUCCESS)
	{
		fprintf(fp, "t minimum = %lf   q = %lf\n", t, q);
		for (i = 0; i < K + 1; i++)
		{
			fprintf(fp, "%lf\n", weights[i]);
		}

	}
	else
	{
		fprintf(fp, "time was not found\n");
	}


	fclose(fp);
}

void initWeights(double* weights, int K)
{
	int i;
	for (i = 0; i < K + 1; i++)
	{
		weights[i] = 0;
	}
}

void allocateCpuMemory(int N, int K, Points* points, double** weights)
{
	points->coordinantes = (double*)malloc((K)*(N) * sizeof(double));
	points->velocity = (double*)malloc((K)*(N) * sizeof(double));
	points->group = (int*)malloc((N) * sizeof(int));
	*weights = (double*)malloc((K + 1) * sizeof(double));
}

void freeCpuMemory(Points* points, double* weights)
{
	free(points->coordinantes);
	free(points->velocity);
	free(points->group);
	free(weights);

}

void dismissUnemployedProcesses(int myCurrentJobIndex, int numOfJobs, int myId)
{
	if (myCurrentJobIndex >= numOfJobs)
	{// will dismiss all the slaves that dont have a job
		printf("myId %d exit the program\n", myId);
		fflush(NULL);
		MPI_Finalize();
		exit(EXIT_SUCCESS);
	}
}


void sendMessageToAllSlaves(int numOfWorkingSlaves, int message, int tag)
{	//send a massege to all relevant slaves
	int i, slaveId;
	for (i = 0; i < numOfWorkingSlaves; i++)
	{
		slaveId = i + 1;
		MPI_Send(&message, 1, MPI_INT, slaveId, tag, MPI_COMM_WORLD);
	}
}


void broadcastParameters(int* N, int* K, double* dt, double* tmax, double* a, int* LIMIT, double* QC)
{

	//Brodcast all parameters to all slaves
	MPI_Bcast(N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(dt, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(tmax, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(LIMIT, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(QC, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	//***Another option is to create one Struct that will contain all parameters, pack it, and send it in one piece.
	//I have measured time and it was pretty much the same so i decided to leave it as is.
}

void broadcastPoints(Points* points, int N, int K)
{
	//broadcast all points to slaves
	MPI_Bcast(points->coordinantes, N*K, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(points->velocity, N*K, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(points->group, N, MPI_INT, MASTER, MPI_COMM_WORLD);
}