mpicc task.c
mpirun -np 6 --hostfile ompihosts ./a.out 1 2 3 10
