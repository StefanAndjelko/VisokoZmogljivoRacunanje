#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "helper_cuda.h"

#define CUBE_SIZE 16

__global__ void calculate_next_step(float *u_array, float *v_array, float *new_u_array, float *new_v_array, int dim, int size, float delta_time, float f, float k, float d_u, float d_v)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    int r = row;
    while (r < dim)
    {
        int c = column;
        while (c < dim)
        {
            int idx = r * dim + c;
            int up_idx = ((r - 1 + dim) % dim) * dim + c;
            int down_idx = ((r + 1) % dim) * dim + c;
            int left_idx = r * dim + ((c - 1 + dim) % dim);
            int right_idx = r * dim + ((c + 1) % dim);
            float u_grad = u_array[down_idx] + u_array[up_idx] + u_array[left_idx] + u_array[right_idx] - 4.0f * u_array[idx];
            float v_grad = v_array[down_idx] + v_array[up_idx] + v_array[left_idx] + v_array[right_idx] - 4.0f * v_array[idx];
            new_u_array[idx] = u_array[idx] + delta_time * (-u_array[idx] * (v_array[idx] * v_array[idx]) + f * (1.0f - u_array[idx]) + d_u * u_grad);
            new_v_array[idx] = v_array[idx] + delta_time * (u_array[idx] * (v_array[idx] * v_array[idx]) - (f + k) * v_array[idx] + d_v * v_grad);
    
            
            c += gridDim.x * blockDim.x;
        }
        r += gridDim.y * blockDim.y;
    }
}

int main(int argc, char *[] argv) 
{
    float *
    checkCudaErrors(cudaMalloc((void **)&device_histogram, 256 * sizeof(int)));

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
}