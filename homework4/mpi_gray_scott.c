#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define COLOR_CHANNELS 3

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void initialize_board(float *u_array, float *v_array, int dim, float u_value, float v_value)
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

void calculate_next_step(float *u_buffer, float *v_buffer, float *new_u_buffer, float *new_v_buffer, float *u_top, float *u_bot, float *v_top, float *v_bot, int height, int width, float delta_time, float d_u, float d_v, float f, float k)
{
    for (int row = 0; row < height; row++)
    {
        for (int column = 0; column < width; column++)
        {
            if (row == 0)
            {
                int idx = row * width + column;
                int left_idx = row * width + ((column - 1 + width) % width);
                int right_idx = row * width + ((column + 1) % width);
                int down_idx = (row + 1) * width + column;

                float u_grad = u_buffer[down_idx] + u_top[column] + u_buffer[left_idx] + u_buffer[right_idx] - 4.0f * u_buffer[idx];
                float v_grad = v_buffer[down_idx] + v_top[column] + v_buffer[left_idx] + v_buffer[right_idx] - 4.0f * v_buffer[idx];

                new_u_buffer[idx] = u_buffer[idx] + delta_time * (-u_buffer[idx] * (v_buffer[idx] * v_buffer[idx]) + f * (1.0f - u_buffer[idx]) + d_u * u_grad);
                new_v_buffer[idx] = v_buffer[idx] + delta_time * (u_buffer[idx] * (v_buffer[idx] * v_buffer[idx]) - (f + k) * v_buffer[idx] + d_v * v_grad);
            }
            else if (row == height - 1)
            {
                int idx = row * width + column;
                int left_idx = row * width + ((column - 1 + width) % width);
                int right_idx = row * width + ((column + 1) % width);
                int up_idx = (row - 1) * width + column;

                float u_grad = u_bot[column] + u_buffer[up_idx] + u_buffer[left_idx] + u_buffer[right_idx] - 4.0f * u_buffer[idx];
                float v_grad = v_bot[column] + v_buffer[up_idx] + v_buffer[left_idx] + v_buffer[right_idx] - 4.0f * v_buffer[idx];

                new_u_buffer[idx] = u_buffer[idx] + delta_time * (-u_buffer[idx] * (v_buffer[idx] * v_buffer[idx]) + f * (1.0f - u_buffer[idx]) + d_u * u_grad);
                new_v_buffer[idx] = v_buffer[idx] + delta_time * (u_buffer[idx] * (v_buffer[idx] * v_buffer[idx]) - (f + k) * v_buffer[idx] + d_v * v_grad);
            }
            else
            {
                int idx = row * width + column;
                int left_idx = row * width + ((column - 1 + width) % width);
                int right_idx = row * width + ((column + 1) % width);
                int up_idx = (row - 1) * width + column;
                int down_idx = (row + 1) * width + column;

                float u_grad = u_buffer[down_idx] + u_buffer[up_idx] + u_buffer[left_idx] + u_buffer[right_idx] - 4.0f * u_buffer[idx];
                float v_grad = v_buffer[down_idx] + v_buffer[up_idx] + v_buffer[left_idx] + v_buffer[right_idx] - 4.0f * v_buffer[idx];

                new_u_buffer[idx] = u_buffer[idx] + delta_time * (-u_buffer[idx] * (v_buffer[idx] * v_buffer[idx]) + f * (1.0f - u_buffer[idx]) + d_u * u_grad);
                new_v_buffer[idx] = v_buffer[idx] + delta_time * (u_buffer[idx] * (v_buffer[idx] * v_buffer[idx]) - (f + k) * v_buffer[idx] + d_v * v_grad);
            }
        }
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

int main(int argc, char* argv[])
{
    int n = 256;
    int steps = 5000;
    float delta_time = 1.0;
    float d_u = 0.16f;
    float d_v = 0.08f;
    float f = 0.06;
    float k = 0.062f;

    int procs, myid;	
    int mystart, myend, myrows;

    float *u_array = (float *) malloc(n * n * sizeof(float));
    float *v_array = (float *) malloc(n * n * sizeof(float));

    initialize_board(u_array, v_array, n, 0.75f, 0.25f);

    // printf("WORKS1, %d", myid);

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);	

    // printf("WORKS2, %d", myid);

    mystart = n / procs * myid;
	myend = n / procs * (myid + 1);
	myrows = n / procs;

    float *u_top = (float*)malloc(n * sizeof(float));
	float *u_bot = (float*)malloc(n * sizeof(float));
    float *v_top = (float*)malloc(n * sizeof(float));
	float *v_bot = (float*)malloc(n * sizeof(float));

    float *u_buffer = (float *) malloc(n * myrows * sizeof(float));
    float *v_buffer = (float *) malloc(n * myrows * sizeof(float));
    float *new_u_buffer = (float *) malloc(n * myrows * sizeof(float));
    float *new_v_buffer = (float *) malloc(n * myrows * sizeof(float));

    // printf("WORKS3, %d", myid);


    MPI_Scatter(u_array, myrows * n, MPI_FLOAT, 
        u_buffer, myrows * n, MPI_FLOAT, 
        0, MPI_COMM_WORLD);
    MPI_Scatter(v_array, myrows * n, MPI_FLOAT, 
        v_buffer, myrows * n, MPI_FLOAT, 
        0, MPI_COMM_WORLD);

    // printf("WORKS4, %d", myid);


    for (int i = 0; i < steps; i++)
    {
		MPI_Sendrecv(u_buffer, n, MPI_FLOAT, (myid + procs - 1) % procs, 0,
					 u_bot, n, MPI_FLOAT, (myid + 1) % procs, 0,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        MPI_Sendrecv(v_buffer, n, MPI_FLOAT, (myid + procs - 1) % procs, 0,
					 v_bot, n, MPI_FLOAT, (myid + 1) % procs, 0,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		MPI_Sendrecv(u_buffer + (myrows- 1) * n, n, MPI_FLOAT, (myid + 1) % procs, 1,
					 u_top, n, MPI_FLOAT, (myid + procs - 1) % procs, 1, 
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        MPI_Sendrecv(v_buffer + (myrows - 1) * n, n, MPI_FLOAT, (myid + 1) % procs, 1,
					 v_top, n, MPI_FLOAT, (myid + procs - 1) % procs, 1, 
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // printf("WORKS5, %d", myid);


		calculate_next_step(u_buffer, v_buffer, new_u_buffer, new_v_buffer, u_top, u_bot, v_top, v_bot, myrows, n, delta_time, d_u, d_v, f, k);
        // printf("WORKS6, %d", myid);

        float *temp_pointer = u_buffer;
        u_buffer = new_u_buffer;
        new_u_buffer = temp_pointer;
        temp_pointer = v_buffer;
        v_buffer = new_v_buffer;
        new_v_buffer = temp_pointer;
	}

    MPI_Gather(u_buffer, myrows * n, MPI_FLOAT, 
        u_array, myrows * n, MPI_FLOAT, 
        0, MPI_COMM_WORLD);
    MPI_Gather(v_buffer, myrows * n, MPI_FLOAT, 
        v_array, myrows * n, MPI_FLOAT, 
        0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        unsigned char *result_image = (unsigned char *)malloc(COLOR_CHANNELS * n * n * sizeof(unsigned char));
        render_to_image(result_image, v_array, n);

        if (!stbi_write_png("Result.png", n, n, 3, result_image, n * 3)) {
            printf("Failed to save image %s\n", "Result.png");
            stbi_image_free(result_image);
            return 1;
        }

        printf("Completed Gray-Scott simulation in %.2f seconds", 0.0f);

        free(result_image);
        free(u_array);
        free(v_array);
    }   

    MPI_Finalize();
}