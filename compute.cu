#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>
#include "cuda.h"
#include <stdio.h>

__global__ void compute_accel(vector3* values, vector3** accels, vector3* dVel, vector3* dPos, double* dMass) {
	int x = blockIdx.x * blockDim.y + threadIdx.x;
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
}

__global__ void add_accel(vector3* values, vector3** accels, vector3* dVel, vector3* dPos, double* dMass) {
	int x = blockIdx.x * blockDim.y + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NUMENTITIES || y >= NUMENTITIES) {
		return;
	}
	if (y != 0) {
		return; // only execute once per object
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

}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute_old() {
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
	free(accels);
	free(values);
}

void compute() {
	//d_hvel and d_hpos hold the hVel and hPos variables on the GPU
	vector3 *d_vel, *d_pos;
	double *d_mass;
	vector3* d_value;
	vector3** d_accel;

	cudaMalloc((void**) &d_vel, (sizeof(vector3) * NUMENTITIES));
	cudaMalloc((void**) &d_pos, (sizeof(vector3) * NUMENTITIES));
	cudaMalloc((void**) &d_mass, (sizeof(double) * NUMENTITIES));

	//Copy memory from the host onto the GPU
	cudaMemcpy(d_vel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	//Allocate space on the GPU for these variables
	cudaMallocManaged((void**) &d_value, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMallocManaged((void**) &d_accel, sizeof(vector3*) * NUMENTITIES);

	//Determine number of blocks that we should be running
	dim3 threadsPerBlock(16, 16, 1);
	dim3 numBlocks((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16, 1);

	compute_accel<<<numBlocks, threadsPerBlock>>>(d_value, d_accel, d_vel, d_pos, d_mass);
	cudaDeviceSynchronize();

	add_accel<<<numBlocks, threadsPerBlock>>>(d_value, d_accel, d_vel, d_pos, d_mass);
	cudaDeviceSynchronize();

	//Copy the results back to the device
	cudaMemcpy(hVel, d_vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos, d_pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(d_mass);
	cudaFree(d_vel);
	cudaFree(d_pos);
	cudaFree(d_value);
	cudaFree(d_accel);
}
