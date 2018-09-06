#!/bin/bash

cant="1000000 2000000 3000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000"

gcc -Wall -o Quicksort_Pthreads Quicksort_Pthreads.c -lm -g -fopenmp
gcc -Wall -o Quicksort_OpenMP Quicksort_OpenMP.c -lm -g -fopenmp

for c in $cant 
do
	echo "Pthreads $c"
	./Quicksort_Pthreads $c
	echo "OpenMP $c"
	./Quicksort_OpenMP $c
done
