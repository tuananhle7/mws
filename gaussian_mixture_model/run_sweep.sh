#!/bin/bash

for seed in 1 2 3 4 5 6 7 8 9 10
do
  for num_particles in 2 5 10 20 50
  do
    for algorithm in vimco rws
    do
      sbatch run.sh $seed $num_particles $algorithm
    done
  done
done
