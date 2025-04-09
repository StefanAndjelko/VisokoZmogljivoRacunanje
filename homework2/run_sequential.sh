#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=sample_out.log


gcc -O2 -lm sequential_histogram_equalization.c -o sequential || exit 1

RESOLUTIONS=("720x480" "1024x768" "1920x1200" "3840x2160" "7680x4320")

for res in "${RESOLUTIONS[@]}"; do
    for i in {1..5}; do
        srun ./sequential "./test_images/${res}.png" >> sequential_output.log
    done
done