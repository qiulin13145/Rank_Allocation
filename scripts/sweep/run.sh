#!/bin/bash

# Loop to run a command 10 times
for i in {1..4}
do
   # Replace 'your_command_here' with the command you want to run
   sbatch sweep_60m_fourier_low_rank.sh
done