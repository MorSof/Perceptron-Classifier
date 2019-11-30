
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "main.h"
#include "device_launch_parameters.h"

cudaError_t allocateCudaMemory(Points* points, int N, int K, Points* dev_points, double** dev_weights, int** dev_results);
cudaError_t calcCoordsWithCuda(Points* points, int N, int K, double* weights, double a, int LIMIT, double t, double proc_dt, int* results, int* Nmiss, Points* dev_points, double* dev_weights, int* dev_results, int* numOfBlocks, int* numOfThreadsPerBlock, int myId);
__global__ void updatePointsCoordinantes(double* dev_coords, double* dev_velocity, int N, int K, double t, double proc_dt);
int checkAllPointsLimitTimesCuda(Points* points, int N, int K, double* dev_coords, int* dev_group, double* dev_weights, double* weights, double a, int LIMIT, int* results, int* dev_results, int numOfBlocks, int numOfThreadsPerBlock, int myId);
int checkAllPointsLimitTimesOMP(Points* points, int N, int K, double* weights, double a, int LIMIT, int* results, int myId);
__global__ void checkAllPointsOneIteration(int N, int K, double* dev_coords, int* dev_group, double* dev_weights, double a, int* dev_results, int myId);
__device__ int calculateWeightFuncCuda(double* dev_coords, double* dev_weights, int K, int id);
void checkError(cudaError_t cudaStatus, void* parameter, const char* errorMessage);
void freeCudaMemory(Points* dev_points, double* dev_weights, int* dev_results);
int countNmiss(int* results, int N, int* minIndex);

//Move points to their new dt coordinantes
cudaError_t calcCoordsWithCuda(Points* points, int N, int K, double* weights, double a, int LIMIT, double t, double proc_dt, int* results, int* Nmiss, Points* dev_points, double* dev_weights, int* dev_results, int* numOfBlocks, int* numOfThreadsPerBlock, int myId)
{
	char errorBuffer[100];
	cudaDeviceProp props;
	cudaError_t cudaStatus;
	cudaGetDeviceProperties(&props, 0);
	const int maxNumOfThreadsPerBlock = props.maxThreadsPerBlock;
	*numOfBlocks = N % maxNumOfThreadsPerBlock == 0 ? N / maxNumOfThreadsPerBlock : (N / maxNumOfThreadsPerBlock) + 1;
	*numOfThreadsPerBlock = N % (*numOfBlocks) == 0 ? N / (*numOfBlocks) : (N / (*numOfBlocks)) + 1;

	cudaStatus = cudaMemcpy(dev_points->coordinantes, points->coordinantes, N*K * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_points->coordinantes, "cudaMemcpy failed!");
	updatePointsCoordinantes << <(*numOfBlocks), (*numOfThreadsPerBlock) >> >(dev_points->coordinantes, dev_points->velocity, N, K, t, proc_dt);
	cudaStatus = cudaMemcpy(points->coordinantes, dev_points->coordinantes, N*K * sizeof(double), cudaMemcpyDeviceToHost);
	checkError(cudaStatus, dev_points->coordinantes, "cudaMemcpy failed!");
	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	return cudaStatus;
}

//Each Cuda thread move a single point to a new location
__global__ void updatePointsCoordinantes(double* dev_coords, double* dev_velocity, int N, int K, double t, double proc_dt)
{
	int i, id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < N*K)
		for (i = 0; i < K; i++)
			dev_coords[i + id * K] = dev_coords[i + id * K] + dev_velocity[i + id * K] * proc_dt;
}

//Check for a success and correct the weights LIMIT times with Cuda + OMP
int checkAllPointsLimitTimesCuda(Points* points, int N, int K, double* dev_coords, int* dev_group, double* dev_weights, double* weights, double a, int LIMIT, int* results, int* dev_results, int numOfBlocks, int numOfThreadsPerBlock, int myId)
{
	int i;
	int Nmiss = 0;
	int firstBrokenIndex;
	char errorBuffer[100];
	cudaError_t cudaStatus;
	for (i = 0; (i < LIMIT); i++)
	{
		cudaStatus = cudaMemcpy(dev_weights, weights, (K + 1) * sizeof(double), cudaMemcpyHostToDevice);
		checkAllPointsOneIteration << <numOfBlocks, numOfThreadsPerBlock >> > (N, K, dev_coords, dev_group, dev_weights, a, dev_results, myId); //Each thread will check if a single point is in the right position and will place the result in the the results array
		cudaStatus = cudaMemcpy(weights, dev_weights, (K + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(results, dev_results, N * sizeof(int), cudaMemcpyDeviceToHost);
		sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		Nmiss = countNmiss(results, N, &firstBrokenIndex); //count the Nmiss from the result array with OMP
		if (Nmiss > 0)
		{	//Faild - fixing the weights 
			fixWeights(K, results[firstBrokenIndex], &(points->coordinantes[firstBrokenIndex*K]), weights, a);
		}
		else
		{	//Success!!
			return Nmiss;
		}
	}
	cudaStatus = cudaMemcpy(weights, dev_weights, (K + 1) * sizeof(double), cudaMemcpyDeviceToHost); //for last iteration
	return Nmiss;
}


__global__ void checkAllPointsOneIteration(int N, int K, double* dev_coords, int* dev_group, double* dev_weights, double a, int* dev_results, int myId)
{//Each thread will check if a single point is in the right position and will place the result in the the results array
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < N)
	{
		int sign = calculateWeightFuncCuda(dev_coords, dev_weights, K, id); //Place one point in the weight function and return the sign
		if ((sign == -1 && dev_group[id] == 1) || (sign == 1 && dev_group[id] == -1))
			dev_results[id] = sign;
		else
			dev_results[id] = 0;
	}
}

__device__ int calculateWeightFuncCuda(double* dev_coords, double* dev_weights, int K, int id)
{//Place one point in the weight function and return the sign
	double sum = 0;
	int i;
	const double FIRST_PARAM = 1;
	sum += FIRST_PARAM * dev_weights[0];

	for (i = 0; i < K; i++)
	{
		sum += dev_coords[i + (id * K)] * dev_weights[i + 1];
	}

	return sum < 0 ? -1 : 1;
}


//count the Nmiss from the result array with OMP
int countNmiss(int* results, int N, int* minIndex)
{
	int i, Nmiss = 0;
	*minIndex = N;
#pragma omp parallel for reduction(+:Nmiss) reduction(min:(*minIndex))
	for (i = 0; i < N; i++)
	{
		if (results[i] != 0)
		{
			Nmiss++;
			if (i < *minIndex)
			{
				*minIndex = i;
			}
		}
	}
	return Nmiss;
}

//Check for a success and correct the weights LIMIT times with OMP
int checkAllPointsLimitTimesOMP(Points* points, int N, int K, double* weights, double a, int LIMIT, int* results, int myId)
{	//Checking for a success
	int i, j;
	int Nmiss = 0;
	int firstBrokenIndex;

	for (i = 0; (i < LIMIT); i++)
	{
		firstBrokenIndex = N;
		j = 0;
#pragma omp parallel for reduction(+:Nmiss) reduction(min:firstBrokenIndex)
		for (j = 0; j < N; j++)
		{
			int sign = calculateWeightFuncOMP(points->coordinantes, weights, K, j);
			if ((sign == -1 && points->group[j] == 1) || (sign == 1 && points->group[j] == -1))
			{
				Nmiss++;
				if (j < firstBrokenIndex)
				{
					firstBrokenIndex = j;
					results[firstBrokenIndex] = sign;
				}
			}
		}

		if ((Nmiss > 0) && (i < LIMIT - 1))
		{	//Faild but there are more iterations in the LIMIT loop
			fixWeights(K, results[firstBrokenIndex], &(points->coordinantes[firstBrokenIndex*K]), weights, a);
			Nmiss = 0;
		}
		else if (Nmiss > 0)
		{	//Faild and there are NOT more iterations in the LIMIT loop
			fixWeights(K, results[firstBrokenIndex], &(points->coordinantes[firstBrokenIndex*K]), weights, a);
		}
		else
		{	//Success!!
			return Nmiss;
		}
	}
	return Nmiss;
}


//Allocate Cuda memory
cudaError_t allocateCudaMemory(Points* points, int N, int K, Points* dev_points, double** dev_weights, int** dev_results)
{
	char errorBuffer[100];
	const char* ERROR_MESSAGE = "cudaMalloc failed!";
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMalloc((void**)&(dev_points->coordinantes), sizeof(double) * N*K);
	checkError(cudaStatus, dev_points->coordinantes, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)&(dev_points->velocity), sizeof(double) * N*K);
	checkError(cudaStatus, dev_points->velocity, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)&(dev_points->group), sizeof(int) * N);
	checkError(cudaStatus, dev_points->group, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)dev_results, sizeof(int) * N);
	checkError(cudaStatus, dev_results, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)dev_weights, sizeof(double) * (K + 1));
	checkError(cudaStatus, dev_weights, ERROR_MESSAGE);

	cudaStatus = cudaMemcpy(dev_points->velocity, points->velocity, N*K * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_points->velocity, "cudaMemcpy failed!");
	cudaStatus = cudaMemcpy(dev_points->group, points->group, N * sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_points->group, "cudaMemcpy failed!");

	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	return cudaStatus;
}

//Free Cuda memory
void freeCudaMemory(Points* dev_points, double* dev_weights, int* dev_results)
{
	cudaFree(dev_points->coordinantes);
	cudaFree(dev_points->velocity);
	cudaFree(dev_points->group);
	cudaFree(dev_weights);
	cudaFree(dev_results);
}

//Check for errors
void checkError(cudaError_t cudaStatus, void* parameter, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		cudaFree(parameter);
	}
}