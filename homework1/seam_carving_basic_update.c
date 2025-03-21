#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <limits.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

void calculate_energy(unsigned char *energy_out, const unsigned char *image_in, int width, int height, int cpp) {
    int mx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int my[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int gx = 0, gy = 0;

            int red_x = 0, green_x = 0, blue_x = 0;
            int red_y = 0, blue_y = 0, green_y = 0;

            for (int ki = -1; ki < 2; ki++) {
                for (int kj = -1; kj < 2; kj++) {
                    int ni = i + ki;
                    int nj = j + kj;
                    if (ni < 0) ni = 0;
                    if (ni >= height) ni = height - 1;
                    if (nj < 0) nj = 0;
                    if (nj >= width) nj = width - 1;

                    const unsigned char *pixel = image_in + (ni * width + nj) * cpp;

                    int weight_x = mx[ki + 1][kj + 1];
                    int weight_y = my[ki + 1][kj + 1];

                    red_x += weight_x * pixel[0];
                    green_x += weight_x * pixel[1];
                    blue_x += weight_x * pixel[2];
                    red_y += weight_y * pixel[0];
                    green_y += weight_y * pixel[1];
                    blue_y += weight_y * pixel[2];
                }
            }

            gx = (double)(red_x + green_x + blue_x) / 3;
            gy = (double)(red_y + green_y + blue_y) / 3;

            int energy = sqrt((double)(gx * gx + gy * gy));
            int index = (i * width + j);

            energy_out[index] = (unsigned char)energy;
        }
    }
    //stbi_write_png("energy.png", width, height, 1, energy_out, width);
}

void update_energy(unsigned char *energy_out, const unsigned char *energy, int width, int height, int cpp, unsigned int * seam_indexes) {

    for (int i = 0; i < height; i++) {
        int out_col = 0;
        for (int j = 0; j < width; j++) {
            int index = (i * width) + j;
            if (index == seam_indexes[i]) {
                continue;
            }
            int index_out = (i * (width - 1) + out_col);
            energy_out[index_out] = energy[index];
            out_col++;
        }
    }
}

void seam_identification(unsigned char * image_out, int width, int height) {
    for (int i = height - 2; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            int index_b = (i + 1) * width + j;
            int index_bl = (i + 1) * width + j - 1;
            int index_br = (i + 1) * width + j + 1;

            if (index_br % width == 0) index_br = index_b;
            if ((index_bl + 1) % width == 0) index_bl = index_b;

            int min_neighbours_index = index_bl;

            if (image_out[index_b] < image_out[min_neighbours_index]) {
                min_neighbours_index = index_b; 
            }
            if (image_out[index_br] < image_out[min_neighbours_index]) {
                min_neighbours_index = index_br; 
            }

            image_out[index] += image_out[min_neighbours_index];
        }
    }
}

void find_vertical_seam(unsigned char * image_out, int width, int height, unsigned int * seam_indexes) {
    int min = INT_MAX;

    for (int j = 0; j < width; j++) {
        if (image_out[j] < min) {
            min = image_out[j];
            seam_indexes[0] = j;
        }
    }

    for (int i = 1; i < height; i++) {
        int index_b = i * width + (seam_indexes[i - 1] % width);
        int index_bl = index_b - 1;
        int index_br = index_b + 1;

        if (index_br % width == 0) index_br = index_b;
        if ((index_bl + 1) % width == 0) index_bl = index_b;

        int min_neighbours_index = index_bl;

        if (image_out[index_b] < image_out[min_neighbours_index]) {
            min_neighbours_index = index_b; 
        }
        if (image_out[index_br] < image_out[min_neighbours_index]) {
            min_neighbours_index = index_br; 
        }
        
        seam_indexes[i] = min_neighbours_index;
    }
}

void remove_vertical_seam(unsigned char * image_out, unsigned char * image_in, int width, int height, int cpp, unsigned int * seam_indexes) {
    for (int i = 0; i < height; i++) {
        int out_col = 0;
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * cpp;
            if (j == seam_indexes[i] % width) {
                continue;
            }
            int index_out = (i * (width - 1) + out_col) * cpp;
            image_out[index_out] = image_in[index];
            image_out[index_out + 1] = image_in[index + 1];
            image_out[index_out + 2] = image_in[index + 2];

            out_col++;
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[255];
    char image_out_name[255];

    snprintf(image_in_name, 255, "%s", argv[1]);
    snprintf(image_out_name, 255, "%s", argv[2]);
    int width_reduction = atoi(argv[3]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL)
    {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_in_name, width, height);

    unsigned char *image_out = image_in;
    double start = omp_get_wtime();
    unsigned char *energy = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    calculate_energy(energy, image_out, width, height, cpp);
    seam_identification(energy, width, height);

    for (int i = 0; i < width_reduction; i++) {
        unsigned int *seam_indexes = (unsigned int *)malloc(height * sizeof(unsigned int));
        find_vertical_seam(energy, width, height, seam_indexes);
        
        unsigned char * updated_energy = (unsigned char *)malloc((width -1) * height * sizeof(unsigned char));
        update_energy(updated_energy, image_out, width, height, cpp,seam_indexes);
        free(energy);
        energy = updated_energy;
        
        
        unsigned char *new_image = (unsigned char *)malloc((width - 1) * height * cpp * sizeof(unsigned char));
        remove_vertical_seam(new_image, image_out, width, height, cpp, seam_indexes);

        free(image_out);
        image_out = new_image;
        width--;

        free(seam_indexes);
    }
    double stop = omp_get_wtime();

    printf(" -> time to copy: %f s\n",stop-start);
    // Write the output image to file
    char image_out_name_temp[255];
    strncpy(image_out_name_temp, image_out_name, 255);
    char *token = strtok(image_out_name_temp, ".");
    char *file_type = NULL;
    while (token != NULL)
    {
        file_type = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);

    // Release the memory
    free(image_in);
    free(image_out);

    return 0;
}