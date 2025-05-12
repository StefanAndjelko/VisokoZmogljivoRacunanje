#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "helper_cuda.h"

#define CUBE_SIZE 16
#define COLOR_CHANNELS 3

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void initialize_grids(float *u_array, float *v_array, int dim, float u_value, float v_value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0)
    {
        int center_start = dim / 2 - dim / 8;
        int center_end = dim / 2 + dim / 8;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                int idx = i * dim + j;
                if (i >= center_start && i < center_end && j >= center_start && j < center_end) {
                    u_array[idx] = u_value;
                    v_array[idx] = v_value;
                } else {
                    u_array[idx] = 1.0f;
                    v_array[idx] = 0.0f;
                }
            }
        }
    }
}

__global__ void calculate_next_step(float *u_array, float *v_array, float *new_u_array, float *new_v_array, int dim, float delta_time, float f, float k, float d_u, float d_v)
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

void render_to_image(unsigned char *image, float *V, int dim) {
    float v_min = V[0];
    float v_max = V[0];

    // First pass: find min and max
    for (int i = 1; i < dim * dim; ++i) {
        if (V[i] < v_min) v_min = V[i];
        if (V[i] > v_max) v_max = V[i];
    }

    float range = v_max - v_min;
    if (range == 0) range = 1.0f; // prevent division by zero

    // Second pass: normalize and convert to grayscale
    for (int i = 0; i < dim * dim; ++i) {
        float v_normalized = (V[i] - v_min) / range;
        unsigned char value = (unsigned char)(v_normalized * 255.0f);
        image[i * COLOR_CHANNELS + 0] = value;
        image[i * COLOR_CHANNELS + 1] = value;
        image[i * COLOR_CHANNELS + 2] = value;
    }
}

void float_print(float *array, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f, ", array[i * n + j]);
        }
        printf("\n");
    }    
}

void char_print(unsigned char *array, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d, ", array[i * n + j]);
        }
        printf("\n");
    }    
}

int main(int argc, char *argv[])
{
    int n = 1024;
    int steps = 5000;
    float delta_time = 1.0;
    float d_u = 0.16f;
    float d_v = 0.08f;
    float f = 0.06;
    float k = 0.062f;


    float *host_u_array = (float*) malloc(n * n * sizeof(float));
    float *host_v_array = (float*) malloc(n * n * sizeof(float));
    float *host_new_u_array = (float*) malloc(n * n * sizeof(float));
    float *host_new_v_array = (float*) malloc(n * n * sizeof(float));

    float *device_u_array, *device_v_array, *device_new_u_array, *device_new_v_array;

    checkCudaErrors(cudaMalloc((void **)&device_u_array, n * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&device_v_array, n * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&device_new_u_array, n * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&device_new_v_array, n * n * sizeof(float)));

    dim3 blockSize(CUBE_SIZE, CUBE_SIZE);
    dim3 gridSize((n + CUBE_SIZE - 1) / CUBE_SIZE, (n + CUBE_SIZE - 1) / CUBE_SIZE);
    initialize_grids<<<1, 32>>>(device_u_array, device_v_array, n, 0.75f, 0.25f);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(host_u_array, device_u_array, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_v_array, device_v_array, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    // printf("U:\n");
    // float_print(host_u_array, n);
    // printf("V:\n");
    // float_print(host_v_array, n);
    unsigned char *init_image = (unsigned char*) malloc(3 * n * n * sizeof(unsigned char));

    render_to_image(init_image, host_u_array, n);

    if (!stbi_write_png("InitImage.png", n, n, 3, init_image, n * 3)) {
        printf("Failed to save image %s\n", "InitImage.png");
        stbi_image_free(init_image);
        return 1;
    }


    for (int i = 0; i < steps; i++)
    {
        calculate_next_step<<<gridSize, blockSize>>>(device_u_array, device_v_array, device_new_u_array, device_new_v_array, n, delta_time, f, k, d_u, d_v);
        checkCudaErrors(cudaGetLastError());
        float *temp_pointer = device_u_array;
        device_u_array = device_new_u_array;
        device_new_u_array = temp_pointer;
        temp_pointer = device_v_array;
        device_v_array = device_new_v_array;
        device_new_v_array = temp_pointer;
    }

    checkCudaErrors(cudaMemcpy(host_u_array, device_u_array, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_v_array, device_v_array, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_v_array));
    checkCudaErrors(cudaFree(device_u_array));
    checkCudaErrors(cudaFree(device_new_u_array));
    checkCudaErrors(cudaFree(device_new_v_array));

    // printf("RESULT U:\n");
    // float_print(host_u_array, n);
    // printf("RESULT V:\n");
    // float_print(host_v_array, n);

    unsigned char *result_image = (unsigned char *)malloc(COLOR_CHANNELS * n * n * sizeof(unsigned char));
    render_to_image(result_image, host_v_array, n);

    free(host_u_array);
    free(host_v_array);
    free(host_new_u_array);
    free(host_new_v_array);

    if (!stbi_write_png("Result.png", n, n, 3, result_image, n * 3)) {
        printf("Failed to save image %s\n", "Result.png");
        stbi_image_free(result_image);
        return 1;
    }

    return 0;
}