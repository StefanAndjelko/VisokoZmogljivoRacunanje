#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --gpus=1
#SBATCH --output=sample_out.log

module load CUDA
nvcc  -diag-suppress 550 -O2 -lm adv_parallel_hist_eq.cu -o adv
RESOLUTIONS=("720x480" "1024x768" "1920x1200" "3840x2160" "7680x4320")
THREAD_BLOCKS=(32 128 160 256 512 1024)
REPETITIONS=10

# Run tests
for threads in "${THREAD_BLOCKS[@]}"; do
  for res in "${RESOLUTIONS[@]}"; do
    for ((i=1; i<=REPETITIONS; i++)); do
      echo "Running $res with $threads threads (attempt $i)"
      srun adv "./test_images/${res}.png" $threads >> adv_out.log
    done
  done
done