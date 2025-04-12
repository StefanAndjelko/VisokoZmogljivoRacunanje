#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --gpus=1
#SBATCH --output=sample_out.log

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm parallel_hist_eq.cu -o parallel
RESOLUTIONS=("720x480" "1024x768" "1920x1200" "3840x2160" "7680x4320")

# Run tests
for res in "${RESOLUTIONS[@]}"; do
    echo "Running $res"
    srun ./parallel "./test_images/${res}.png" > sample_out.log
done