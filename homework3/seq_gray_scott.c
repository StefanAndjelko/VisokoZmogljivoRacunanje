#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define CPP 3

void initialize_grids(float *U, float *V, int grid_size, float uCen, float vCen, float uElse, float vElse) {
    int center_start = grid_size / 2 - grid_size / 8;
    int center_end = grid_size / 2 + grid_size / 8;

    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int idx = i * grid_size + j;
            if (i >= center_start && i < center_end && j >= center_start && j < center_end) {
                U[idx] = uCen;
                V[idx] = vCen;
            } else {
                U[idx] = uElse;
                V[idx] = vElse;
            }
        }
    }
}

void gray_scott(float **U_ptr, float **V_ptr, int grid_size, float Du, float Dv, float F, float k, float dt, int steps) {
    float *U = *U_ptr;
    float *V = *V_ptr;
    float *U_next = (float *)malloc(grid_size * grid_size * sizeof(float));
    float *V_next = (float *)malloc(grid_size * grid_size * sizeof(float));

    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < grid_size; ++i) {
            int ip = (i + 1) % grid_size;
            int im = (i - 1 + grid_size) % grid_size;
            for (int j = 0; j < grid_size; ++j) {
                int jp = (j + 1) % grid_size;
                int jm = (j - 1 + grid_size) % grid_size;
                int idx = i * grid_size + j;

                int idx_up = im * grid_size + j;
                int idx_down = ip * grid_size + j;
                int idx_left = i * grid_size + jm;
                int idx_right = i * grid_size + jp;

                float u = U[idx];
                float v = V[idx];
                float lap_u = U[idx_up] + U[idx_down] + U[idx_left] + U[idx_right] - 4 * u;
                float lap_v = V[idx_up] + V[idx_down] + V[idx_left] + V[idx_right] - 4 * v;

                float uvv = u * v * v;

                U_next[idx] = u + dt * (-uvv + F * (1.0f - u) + Du * lap_u);
                V_next[idx] = v + dt * (+uvv - (F + k) * v + Dv * lap_v);
            }
        }

        float *tmpU = U;
        float *tmpV = V;
        U = U_next;
        V = V_next;
        U_next = tmpU;
        V_next = tmpV;
    }

    free(U_next);
    free(V_next);

    *U_ptr = U;
    *V_ptr = V;
}

void render_to_image(unsigned char *image, float *V, int grid_size) {
    for (int i = 0; i < grid_size * grid_size; ++i) {
        float v = V[i];
        if (v < 0) v = 0;
        if (v > 1) v = 1;

        unsigned char value = (unsigned char)(v * 255);
        image[i * CPP + 0] = value; // R
        image[i * CPP + 1] = value; // G
        image[i * CPP + 2] = value; // B
    }
}


int main(int argc, char *argv[])
{

    if (argc < 12)
    {
        printf("USAGE: sequential grid_size time_step du dv f k uCen vCen uElse vElse steps \n");
        exit(EXIT_FAILURE);
    }

    int grid_size = atoi(argv[1]);
    float time_step = atof(argv[2]);
    float du = atof(argv[3]);
    float dv = atof(argv[4]);
    float f = atof(argv[5]);
    float k = atof(argv[6]);
    float uCen = atof(argv[7]);
    float vCen = atof(argv[8]);
    float uElse = atof(argv[9]);
    float vElse = atof(argv[10]);
    int steps = atoi(argv[11]);

    char szImage_out_name[500];

    snprintf(szImage_out_name, 500, "out_%dx%d_t%.3f_du%.3f_dv%.3f_f%.3f_k%.3f_steps%d.png", grid_size, grid_size, time_step, du, dv, f, k, steps);

    const size_t datasize = grid_size * grid_size * CPP * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    clock_t program_start = clock();

    float *U = (float *)malloc(grid_size * grid_size * sizeof(float));
    float *V = (float *)malloc(grid_size * grid_size * sizeof(float));

    initialize_grids(U, V, grid_size, uCen, vCen, uElse, vElse);
    gray_scott(&U, &V, grid_size, du, dv, f, k, time_step, steps);
    render_to_image(h_imageOut, V, grid_size);

    clock_t program_end = clock();
    double execution_time = (double)(program_end - program_start) / CLOCKS_PER_SEC;

    free(U);
    free(V);

    if (!stbi_write_png(szImage_out_name, grid_size, grid_size, CPP, h_imageOut, grid_size * CPP)) {
        printf("Failed to save image %s\n", szImage_out_name);
        stbi_image_free(h_imageOut);
        return 1;
    }

    printf("Saved Gray-Scott as %s\n", szImage_out_name);
    printf("Executed Gray-Scott on a grid %dx%d, for parameters time_step=%.3f, du=%.3f, dv=%.3f, f=%.3f, k=%.3f, uCen=%.3f, vCen=%.3f, uElse=%.3f, vElse=%.3f \n in %.3f (ms)", grid_size, grid_size, time_step, du, dv, f, k, uCen, vCen, uElse, vElse, execution_time * 1000.0);

    stbi_image_free(h_imageOut);

    return 0;
}


// gcc seq_gray_scott.c -o seq -lm
// ./seq 256 1.0 0.16 0.08 0.060 0.062 0.75 0.25 1.0 0.0 5000
