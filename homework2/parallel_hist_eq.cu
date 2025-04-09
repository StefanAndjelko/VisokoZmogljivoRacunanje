// module load CUDA/11.1.1-GCC-10.2.0

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0
#define BLOCK_SIZE 256

__device__ void to_YUV_color_space(unsigned char *image_in, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (tid < width * height) {
        int idx = tid * 3;
        
        unsigned char r = image_in[idx];
        unsigned char g = image_in[idx + 1];
        unsigned char b = image_in[idx + 2];
        
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        float u = -0.168736f *r + -0.331264f * g + 0.5f * b + 128.0f;
        float v = 0.5f * r + -0.418688f * g + -0.081312f * b + 128.0f;
        
        image_in[idx] = (unsigned char)fminf(fmaxf(y, 0.0f), 255.0f);
        image_in[idx + 1] = (unsigned char)fminf(fmaxf(u, 0.0f), 255.0f);
        image_in[idx + 2] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
        
        tid += gridDim.x * blockDim.x;
    }
}

__device__ void luminance_histogram(unsigned char *image_in, int width, int height, int *histogram)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < width * height)
    {
        int idx = tid * 3;
        unsigned char luminance = (int)image_in[idx];
        atomicAdd(&histogram[luminance], 1);
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void histogram_normalization(unsigned char *image_in, int width, int height, int *histogram)
{
    to_YUV_color_space(image_in, width, height);
    luminance_histogram(image_in, width, height, histogram);
}

__global__ void cumulative_histogram(int* histogram, int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        for (int i = 1; i < len; i++)
        {
            histogram[i] += histogram[i - 1];
        }
    }
}

__global__ void new_luminance(unsigned char* luminance, int *c_hist, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int min_cdf;
    if (tid == 0)
    {
        min_cdf = 0;
        for (int i = 0; i < 256; i++)
        {
            if (c_hist[i] > 0)
            {
                min_cdf = c_hist[i];
                break;
            }
        }
    }
    __syncthreads();
    if (tid < 256)
    {
        float scale = 255.0f / (width * height - min_cdf);
        luminance[tid] = (c_hist[tid] - min_cdf) * scale;
    }
}

__device__ void set_new_luminances(unsigned char *image_in, int width, int height, unsigned char *luminances)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < width * height)
    {
        int idx = tid * 3;
        int l_value = (int) image_in[idx];
        image_in[idx] = luminances[l_value];
        tid += gridDim.x * blockDim.x;
    }
}

__device__ void to_RGB_color_space(unsigned char *image_in, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < width * height)
    {
        int idx = tid * 3;
        double y = image_in[idx];
        double u = image_in[idx + 1] - 128; 
        double v = image_in[idx + 2] - 128;

        double r = y + v * 1.402;
        double g = y - u * 0.344136 - v * 0.714136;
        double b = y + u * 1.772;

        image_in[idx] = (unsigned char)fminf(fmaxf(r, 0.0f), 255.0f);
        image_in[idx + 1] = (unsigned char)fminf(fmaxf(g, 0.0f), 255.0f);
        image_in[idx + 2] = (unsigned char)fminf(fmaxf(b, 0.0f), 255.0f);

        tid += gridDim.x * blockDim.x;
    }
}

__global__ void generate_final_image(unsigned char *in_image, int width, int height, unsigned char *luminances)
{
    set_new_luminances(in_image, width, height, luminances);
    to_RGB_color_space(in_image, width, height);
}

void print_image(unsigned char *arr, int len)
{
    for (int i = 0; i < len; i+=3)
    {
        printf("(%d, %d, %d), ", arr[i], arr[i + 1], arr[i + 2]);
    }
    printf("\n");
}

void print_array(int *arr, int len)
{
    for (int i = 0; i < len; i++)
    {
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

void print_array(unsigned char *arr, int len)
{
    for (int i = 0; i < len; i++)
    {
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[])
{

    char *image_in_name = argv[1];
    int width, height, cpp;
    unsigned char *host_image = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    int image_size = width * height;

    unsigned char *device_image;
    checkCudaErrors(cudaMalloc((void **)&device_image, image_size * 3 * sizeof(unsigned char)));

    // HISTOGRAM
    int *host_histogram = (int *)calloc(256, sizeof(int));
    int *device_histogram;
    checkCudaErrors(cudaMalloc((void **)&device_histogram, 256 * sizeof(int)));

    // LUMINANCE
    unsigned char *host_luminance = (unsigned char *)calloc(256, sizeof(unsigned char));
    unsigned char *device_luminance;
    checkCudaErrors(cudaMalloc((void **)&device_luminance, 256 * sizeof(unsigned char)));

    checkCudaErrors(cudaMemcpy(device_image, host_image, image_size * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_histogram, host_histogram, 256 * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_luminance, host_luminance, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((image_size + blockSize.x - 1)/blockSize.x);

    // Measure compute time
    float elapsedTime;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    histogram_normalization<<<gridSize, blockSize>>>(device_image, width, height, device_histogram);
    cumulative_histogram<<<1, 32>>>(device_histogram, 256);
    new_luminance<<<1, 256>>>(device_luminance, device_histogram, width, height);
    generate_final_image<<<gridSize, blockSize>>>(device_image, width, height, device_luminance);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

    // Copy solution back to host
    checkCudaErrors(cudaMemcpy(host_image, device_image, image_size * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_histogram, device_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_luminance, device_luminance, 256 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_image));
    checkCudaErrors(cudaFree(device_histogram));
    checkCudaErrors(cudaFree(device_luminance));

    if (!stbi_write_png("basic_parallel.png", width, height, 3, host_image, width * 3)) {
        printf("Failed to save image %s\n", "pasic_parallel.png");
        stbi_image_free(host_image);
        return 1;
    }

    printf("Saved modified image as %s\n", "basic_parallel.png");
    printf("Performed histogram normalization on image in %.2f(ms)", elapsedTime);

    printf("\n");
    free(host_image);
    free(host_histogram);
    free(host_luminance);

    return 0;
}
