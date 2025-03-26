#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0

void print_arr(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%d, ", a[i]);
    }
    printf("\n");
}

void print_arr_double(double *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%.2f, ", a[i]);
    }
    printf("\n");
}

double *calculate_energy(unsigned char *image, int n, int m)
{
    double *energy_arr = malloc((n * m) * sizeof(double));

    int thread_num = omp_get_max_threads();

    #pragma omp parallel for
    for (int i = 0; i < n * m * 3; i+= 3) {
        int row_ix = (i / 3) / m;
        int col_ix = (i / 3) % m;

        int col_x1 = col_ix + 1;
        col_x1 = (col_x1 >= m)? (m - 1) : ((col_x1 < 0)? 0 : col_x1);
        int col_x2 =col_ix - 1;
        col_x2 = (col_x2 >= m)? (m - 1) : ((col_x2 < 0)? 0 : col_x2);
        int row_y1 = row_ix - 1;
        row_y1 = (row_y1 >= n)? (n - 1) : ((row_y1 < 0)? 0 : row_y1);
        int row_y2 = row_ix + 1;
        row_y2 = (row_y2 >= n)? (n - 1) : ((row_y2< 0)? 0 : row_y2);


        int g_x_r = 0;
        int g_x_g = 0;
        int g_x_b = 0;

        int g_y_r = 0;
        int g_y_g = 0;
        int g_y_b = 0;

        int g_x = 0;
        int g_y = 0;

        for (int k = -1; k <= 1; k++) 
        {
            int row_x = row_ix + k;
            row_x = (row_x >= n)? (n - 1) : ((row_x < 0)? 0 : row_x);
            int col_y = col_ix + k;
            col_y = (col_y >= m)? (m - 1) : ((col_y < 0)? 0 : col_y);

            g_x_r = g_x_r + image[(row_x * m + col_x1) * 3] - image[(row_x * m + col_x2) * 3];
            g_y_r = g_y_r + image[(row_y1 * m + col_y) * 3] - image[(row_y2 * m + col_y) * 3];
            g_x_g = g_x_g + image[(row_x * m + col_x1) * 3 + 1] - image[(row_x * m + col_x2) * 3 + 1];
            g_y_g = g_y_g + image[(row_y1 * m + col_y) * 3 + 1] - image[(row_y2 * m + col_y) * 3 + 1];
            g_x_b = g_x_b + image[(row_x * m + col_x1) * 3 + 2] - image[(row_x * m + col_x2) * 3 + 2];
            g_y_b = g_y_b + image[(row_y1 * m + col_y) * 3 + 2] - image[(row_y2 * m + col_y) * 3 + 2];
        }

        double energy_r = sqrt((double)(g_x_r * g_x_r + g_y_r * g_y_r));
        double energy_g = sqrt((double)(g_x_g * g_x_g + g_y_g * g_y_g));
        double energy_b = sqrt((double)(g_x_b * g_x_b + g_y_b * g_y_b));

        energy_arr[i / 3] = (double)(energy_r + energy_g + energy_b) / 3;
    }

    return energy_arr;
}

double min3(double a, double b, double c) {
    double min_val = a;

    if (b < min_val) {
        min_val = b;
    }
    if (c < min_val) {
        min_val = c;
    }

    return min_val;
}

double min2(double a, double b) {
    double min_val = a;

    if (b < min_val) {
        min_val = b;
    }

    return min_val;
}

void cumulative_energy(double *array, int n, int m, int thread_num)
{
    // printf("threads: %d\n", thread_num);
    int tri_m = m / thread_num;
    int tri_n = (tri_m + 1) / 2;

    // printf("n: %d, m: %d\n", tri_n, tri_m);
    fflush(stdout);
    for (int base_row = 0; base_row <= n; base_row += tri_n) {
        // printf("Rightside up:\n");
        #pragma omp parallel for
        for (int start_col = 0; start_col <= m; start_col += tri_m) {

            for (int r = 0; r < tri_n; r++) {
                int row_index = base_row + r;
                int start = start_col + r;
                int end = start_col + tri_m - r - 1;

                for (int c = start; c <= end; c++) {
                    if (c < 0 || c >= m || row_index < 0 || row_index >= n || row_index - 1 < 0)
                        continue;
                    int pixel_index = row_index * m + c;
                    // printf("(%d, %d)\n", row_index, c);

                    // printf("%.2f, ", array[pixel_index]);
                    if (c == 0) 
                    {
                        array[row_index * m + c] += min2(array[(row_index - 1) * m + c], array[(row_index - 1) * m + (c + 1)]);
                    }
                    else if (c == m - 1)
                    {
                        array[row_index * m + c] += min2(array[(row_index - 1) * m + c], array[(row_index - 1) * m + (c - 1)]);
                    }
                    else 
                    {
                        array[row_index * m + c] += min3(array[(row_index - 1) * m + c - 1], array[(row_index - 1) * m + c], array[(row_index - 1) * m + (c + 1)]);
                    }
                }
            }
            // printf("\n");
        }

        // printf("Upside down:\n");
        #pragma omp parallel for
        for (int start_col = - 1; start_col < m; start_col += tri_m)
        {
            int shift = 0;
            // int row_shift = 0;
            for (int r = 1; r < tri_n; r++)
            {
                int row_index = base_row + r;
                int start = start_col - shift;
                int end = start_col + shift + 1;

                for (int c = start; c <= end; c++)
                {
                    if (c < 0 || c >= m || row_index < 0 || row_index >= n || row_index || row_index - 1 < 0)
                        continue;

                    // printf("(%d, %d)\n", row_index, c);

                    if (c == 0) 
                    {
                        array[row_index * m + c] += min2(array[(row_index - 1) * m + c], array[(row_index - 1) * m + (c + 1)]);
                    }
                    else if (c == m - 1)
                    {
                        array[row_index * m + c] += min2(array[(row_index - 1) * m + c], array[(row_index - 1) * m + (c - 1)]);
                    }
                    else 
                    {
                        array[row_index * m + c] += min3(array[(row_index - 1) * m + c - 1], array[(row_index - 1) * m + c], array[(row_index - 1) * m + (c + 1)]);
                    }
                    // printf("%.2f, ", array[row_index * m + c]);
                }
                shift++;
            }
            // printf("\n");
        }
    }
}

void simple_cumulative_energy(double *array, int n, int m)
{
    for (int i = n - 2; i >= 0; i--) 
    {
        #pragma omp parallel for
        for (int j = 0; j < m; j++)
        {
            // printf("(%d, %d)\n", i, j);

            if (j == 0) 
            {
                array[i * m + j] += min2(array[(i + 1) * m + j], array[(i + 1) * m + (j + 1)]);
            }
            else if (j == m - 1)
            {
                array[i * m + j] += min2(array[(i + 1) * m + j], array[(i + 1) * m + (j - 1)]);
            }
            else 
            {
                array[i * m + j] += min3(array[(i + 1) * m + j - 1], array[(i + 1) * m + j], array[(i + 1) * m + (j + 1)]);
            }
        }
    }
}

int *shortest_path(double *energies, int n, int m)
{
    int *indices = malloc(n * sizeof(int));

    int min_ix = 0;
    double min_energy = energies[0]; 

    for (int col = 1; col < m; col++)
    {
        if (energies[col] < min_energy)
        {
            min_energy = energies[col];
            min_ix = col;
        }
    }

    indices[0] = min_ix;
    for (int row = 1; row < n; row++)
    {
        int prev_col = indices[row - 1];
        int best_col = prev_col; 
        double best_energy = energies[row * m + prev_col];

        if (prev_col > 0 && energies[row * m + (prev_col - 1)] < best_energy)
        {
            best_energy = energies[row * m + (prev_col - 1)];
            best_col = prev_col - 1;
        }

        if (prev_col < m - 1 && energies[row * m + (prev_col + 1)] < best_energy)
        {
            best_energy = energies[row * m + (prev_col + 1)];
            best_col = prev_col + 1;
        }

        indices[row] = best_col;
    }

    return indices;
}


unsigned char *update_image(unsigned char *image, int n, int m, int *indices)
{
    unsigned char *new_image = malloc(3 * n * (m - 1) * sizeof(unsigned char));

    #pragma omp parallel for
    for (int row = 0; row < n; row++) {
        int counter = row * (m - 1) * 3;
        for (int col = 0; col < m; col++) {
            if (indices[row] == col) 
                continue;

            int src_index = (row * m + col) * 3;
            new_image[counter] = image[src_index];
            new_image[counter + 1] = image[src_index + 1];
            new_image[counter + 2] = image[src_index + 2];
            counter += 3;
        }
    }

    return new_image;
}

double *update_energies(double *energies, int n, int m, int *indices)
{
    double *new_energies = malloc(n * (m - 1) * sizeof(double));

    #pragma omp parallel for
    for (int row = 0; row < n; row++) {
        int counter = row * (m - 1);
        for (int col = 0; col < m; col++) {
            if (indices[row] == col) 
                continue;

            new_energies[counter++] = energies[row * m + col];
        }
    }

    return new_energies;
}

int main(int argc, char *argv[])
{
    char *image_in_name = argv[1];
    int reduce_width = atoi(argv[2]);
    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    int thread_num = omp_get_max_threads();
    printf("Using %d threads\n", thread_num);

    // double *test_arr = malloc(40 * sizeof(double));
    // double *test_arr2 = malloc(40 * sizeof(double));
    // for (int i = 0; i < 40; i++)
    // {
    //     test_arr[i] = (double)(i + 1);
    //     test_arr2[i] = (double)(i + 1);
    // }


    // cumulative_energy(test_arr, 4, 10, thread_num);
    // printf("\n");
    // simple_cumulative_energy(test_arr2, 4, 10);
    // print_arr_double(test_arr, 40);
    // printf("\n");
    // print_arr_double(test_arr2, 40);

    printf("Loaded image of dimensions: (%d, %d)\n", width, height);

    printf("Reducing image by %d columns\n", reduce_width);
    double dt = omp_get_wtime();

    for (int i = 0; i < reduce_width; i++)
    {
        double *energies = calculate_energy(image_in, height, width);
        cumulative_energy(energies, height, width, thread_num);
        int *indices = shortest_path(energies, height, width);
        free(energies);

        unsigned char *new_img = update_image(image_in, height, width, indices);
        free(image_in);
        image_in = new_img;

        width--;
        free(indices);
    }

    dt = omp_get_wtime() - dt;

    if (!stbi_write_png("advanced_test.png", width, height, 3, image_in, width * 3)) {
        printf("Failed to save image %s\n", "advanced_test.png");
        stbi_image_free(image_in);
        return 1;
    }

    printf("Saved modified image as %s\n", "advanced_test.png");
    printf("Image cropped in %.4f seconds,\n", dt);

    stbi_image_free(image_in);

    return 0;
}