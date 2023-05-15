#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
#include "cuda.h"
#include <stdio.h>

__global__ void compute_accel(vector3* values, vector3** accels, vector3* dVel, vector3* dPos, double* dMass) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NUMENTITIES || y >= NUMENTITIES) {
		return;
	}
	if (x == 0) {
		accels[x] = &values[x * NUMENTITIES];
	}
	__syncthreads();
	if (x == y) {
		FILL_VECTOR(accels[x][y], 0, 0, 0);
	} else {
		vector3 distance;
		int k;
		for (k = 0; k<3; k++) {
			distance[k] = dPos[x][k] - dPos[y][k];
		};
		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * dMass[y] / magnitude_sq;
		FILL_VECTOR(accels[x][y], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
	}
	dPos[x][0] = 1000000000.0;

}

__global__ void add_accel(vector3* values, vector3** accels, vector3* dVel, vector3* dPos, double* dMass) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= NUMENTITIES) {
		return;
	}
	vector3 accel_sum = {0, 0, 0};
	int j, k;
	for (j = 0; j < NUMENTITIES; j++) {
		for (k = 0; k < 3; k++) {
			accel_sum[k] += accels[x][j][k];
		}
	}
	//compute the new velocity based on the acceleration and time interval
	//compute the new position based on the velocity and time interval
	for (k = 0; k < 3; k++) {
		dVel[x][k] += accel_sum[k] * INTERVAL;
		dPos[x][k] = dVel[x][k] * INTERVAL;
	}
	dPos[x][0] = 1000000000.0;
}

void compute() {
	//d_hvel and d_hpos hold the hVel and hPos variables on the GPU
	vector3 *dVel, *dPos;
	double *dMass;
	vector3* dValue;
	vector3** dAccel;

	int e = 0;

	e+=cudaMallocManaged(&dVel, (sizeof(vector3) * NUMENTITIES));
	cudaMallocManaged(&dPos, (sizeof(vector3) * NUMENTITIES));
	cudaMallocManaged(&dMass, (sizeof(double) * NUMENTITIES));

	//Copy memory from the host onto the GPU
	cudaMemcpy(dVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dMass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	//Allocate space on the GPU for these variables
	cudaMallocManaged(&dValue, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMallocManaged(&dAccel, sizeof(vector3*) * NUMENTITIES);

	//Determine number of blocks that we should be running
	dim3 threadsPerBlock(16, 16, 1);
	dim3 numBlocks((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16, 1);

	compute_accel<<<numBlocks, threadsPerBlock>>>(dValue, dAccel, dVel, dPos, dMass);
	cudaDeviceSynchronize();

	add_accel<<<(NUMENTITIES + 15) / 16, 16>>>(dValue, dAccel, dVel, dPos, dMass);
	cudaDeviceSynchronize();

	//Copy the results back to the device
	cudaMemcpy(hVel, dVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, dPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(mass, dMass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(dMass);
	cudaFree(dVel);
	cudaFree(dPos);
	cudaFree(dValue);
	cudaFree(dAccel);

	if (e) {
		printf("%d\n", e);
	exit(0);
	}
}
