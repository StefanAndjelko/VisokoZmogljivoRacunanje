#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>

/*
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"
*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0

//__global__ double to_YUV_color_space(const unsigned char *image_in, int width, int height) {
unsigned char *to_YUV_color_space(const unsigned char *image_in, int width, int height) {
    double yuv_conversion_matrix[3][3] = {
        {0.299, 0.587, 0.114}, 
        {-0.168736, -0.331264, 0.5}, 
        {0.5, -0.418688, -0.081312}
    };
    unsigned char *yuv_image = malloc(3 * height * width * sizeof(unsigned char));

    for (int i = 0; i < height * width * 3; i+= 3) {
        double y = image_in[i] * yuv_conversion_matrix[0][0] + image_in[i + 1] * yuv_conversion_matrix[0][1] + image_in[i + 2] * yuv_conversion_matrix[0][2];
        double u = image_in[i] * yuv_conversion_matrix[1][0] + image_in[i + 1] * yuv_conversion_matrix[1][1] + image_in[i + 2] * yuv_conversion_matrix[1][2] + 128.0;
        double v = image_in[i] * yuv_conversion_matrix[2][0] + image_in[i + 1] * yuv_conversion_matrix[2][1] + image_in[i + 2] * yuv_conversion_matrix[2][2] + 128.0;
    
        yuv_image[i] = (unsigned char)round(y < 0 ? 0 : (y > 255 ? 255 : y));
        yuv_image[i + 1] = (unsigned char)round(u < 0 ? 0 : (u > 255 ? 255 : u));
        yuv_image[i + 2] = (unsigned char)round(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    return yuv_image;
}

void to_RGB_color_space(const unsigned char *image_in, unsigned char *image_out, int width, int height) {
    for (int i = 0; i < height * width * 3; i += 3) {
        double y = image_in[i];
        double u = image_in[i + 1] - 128; 
        double v = image_in[i + 2] - 128;

        double r = y + v * 1.402;
        double g = y - u * 0.344136 - v * 0.714136;
        double b = y + u * 1.772;

        image_out[i] = (unsigned char)round(r < 0 ? 0 : (r > 255 ? 255 : r));
        image_out[i + 1] = (unsigned char)round(g < 0 ? 0 : (g > 255 ? 255 : g));
        image_out[i + 2] = (unsigned char)round(b < 0 ? 0 : (b > 255 ? 255 : b));
    }
}

int *calculate_histogram(const unsigned char *yuv_image, int width, int height) {
    int *histogram = (int *)calloc(256, sizeof(int));
    for (int i = 0; i < height * width * 3; i+= 3) {
        histogram[yuv_image[i]]++;
    }

    return histogram;
}

int *accumulate_histogram(int *histogram) {
    int *cumulative_histogram = (int *)malloc(256 * sizeof(int));
    cumulative_histogram[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i];
    }

    return cumulative_histogram;
}

unsigned char *equalize_yuv_image(const unsigned char *yuv_image, int *histogram, int *cumulative_histogram, int width, int height) {
    int min = 0;
    for (int i = 0; i < 256; i++) {
        if (cumulative_histogram[i] != 0) {
            min = cumulative_histogram[i];
            break;
        }
    }

    double *equalized_luminance = (double *)malloc(256 * sizeof(double));
    for (int i = 0; i < 256; i++) {
        equalized_luminance[i] = ((cumulative_histogram[i] - min) / (double) (height * width - min)) * 255.0;
    }

    unsigned char *equalized_image = malloc(3 * height * width * sizeof(unsigned char));
    for (int i = 0; i < height * width * 3; i+= 3) {
        equalized_image[i] = (unsigned char)round(equalized_luminance[yuv_image[i]]);
        equalized_image[i + 1] = yuv_image[i + 1];
        equalized_image[i + 2] = yuv_image[i + 2];
    }
    free(equalized_luminance);

    return equalized_image;
}

/*
__global__ void copy_image(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height, const int cpp)
{

    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (gidx == 0 & gidy == 0)
    {
        printf("DEVICE: START COPY\n");
    }
    for (int i = gidx; i < height; i += blockDim.x * gridDim.x)
    {
        for (int j = gidy; j < width; j += blockDim.y * gridDim.y)
        {
            for (int c = 0; c < cpp; c += 1)
            {
                imageOut[(i * width + j) * cpp + c] = imageIn[(i * width + j) * cpp + c];
            }
        }
    }

}
*/

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("USAGE: sequential input_image \n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    //printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    clock_t program_start = clock();
    unsigned char *image_YUV = to_YUV_color_space(h_imageIn, width, height);
    int *histogram = calculate_histogram(image_YUV, width, height);
    int *cumulative_histogram = accumulate_histogram(histogram);
    unsigned char *equalized_yuv_image = equalize_yuv_image(image_YUV, histogram, cumulative_histogram, width, height);
    to_RGB_color_space(equalized_yuv_image, h_imageOut, width, height);
    clock_t program_end = clock();
    double execution_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;

    /*
    // Setup Thread organization
    dim3 blockSize(16, 16);
    dim3 gridSize((height-1)/blockSize.x+1,(width-1)/blockSize.y+1);
    //dim3 gridSize(1, 1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy image to device and run kernel
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));
    copy_image<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height, cpp);
    checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));
    getLastCudaError("copy_image() execution failed\n");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);
    */

    free(image_YUV);
    free(histogram);
    free(cumulative_histogram);
    free(equalized_yuv_image);
    stbi_image_free(h_imageIn);

    if (!stbi_write_png("test.png", width, height, cpp, h_imageOut, width * cpp)) {
        printf("Failed to save image %s\n", "test.png");
        stbi_image_free(h_imageOut);
        return 1;
    }

    printf("Saved modified image as %s\n", "test.png");
    printf("Image equalized in %.4f seconds,\n", execution_time);

    /*
    // Free device memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));

    // Clean-up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */

    // Free host memory
    stbi_image_free(h_imageOut);


    return 0;
}
