#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>


#define MASTER 0


typedef struct Data {
  int l_;
  int a_;
  int b_;
  int N_;
} Data;


typedef struct Point {
  int x_;
  int y_;
  int r_;
} Point;


Data* getInitialData(char** args) {
  Data* data = calloc(1, sizeof(Data));
  assert(data);
  data->l_ = atoi(args[1]);
  data->a_ = atoi(args[2]);
  data->b_ = atoi(args[3]);
  data->N_ = atoi(args[4]);
  return data;
}


void pointsGeneration(Data* data, int rank, int size) {
  int i;

  double start_time = MPI_Wtime();

  int l = data->l_;
  int a = data->a_;
  int b = data->b_;
  int N = data->N_;
  Point* points = (Point*)calloc(N, sizeof(Point));
  assert(points);

  size_t seed;
  size_t* seeds = (size_t*)calloc(size, sizeof(size_t));
  assert(seeds);
  if (rank == MASTER) {
    srand(time(NULL));
    for (i = 0; i < size; ++i) {
      seeds[i] = rand();
    }
  }
  MPI_Scatter(seeds, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  
  srand(seed);

  for(i = 0; i < N; ++i) {
    points[i].x_ = rand() % l;
    points[i].y_ = rand() % l;
    points[i].r_ = rand() % (a * b);
  }

  int mould_size = l * l * size;
  int* mould = (int*)calloc(mould_size, sizeof(int));
  assert(mould);

  for (i = 0; i < mould_size; ++i) {
    mould[i] = 0;
  }

  for(i = 0; i < N; ++i) {
    int x = points[i].x_;
    int y = points[i].y_;
    int r = points[i].r_;
    ++mould[y * l * size + x * size + r];
  }

  MPI_File f;
  MPI_File_delete("data.bin", MPI_INFO_NULL);
  MPI_File_open(MPI_COMM_WORLD, "data.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

  MPI_Aint intex;
  MPI_Aint ext;
  MPI_Type_get_extent(MPI_INT, &ext, &intex);

  MPI_Datatype view;
  MPI_Type_vector(l, l * size, l * a * size, MPI_INT, &view);
  MPI_Type_commit(&view);

  int row = rank / a;
  int column = rank % a;

  int offset = column * l * size + row * a * mould_size;
  offset *= sizeof(int);
  MPI_File_set_view(f, offset, MPI_INT, view, "native", MPI_INFO_NULL);

  MPI_File_write(f, mould, mould_size, MPI_INT, MPI_STATUS_IGNORE);

  double end_time = MPI_Wtime();

  if (rank == MASTER) {
    double time = end_time - start_time;
    FILE* stats = fopen("stats.txt", "w");
    fprintf(stats, "%d %d %d %d %1.3lfs\n", l, a, b, N, time);
    fclose(stats);
  }

  free(points);
  free(seeds);
  free(mould);
  MPI_Type_free(&view);
  MPI_File_close(&f);

}


int main(int argc, char **argv) {
  Data* data = getInitialData(argv);

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //assert(size == data->a_ * data->b_);
  pointsGeneration(data, rank, size);

  MPI_Finalize();

  free(data);
  return 0;
}
