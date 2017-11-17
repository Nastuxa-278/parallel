mpicc task.c
mpirun -np 6 --hostfile ompihosts ./a.out 10 2 3 100 10 0.25 0.25 0.25 0.25
