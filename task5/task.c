#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <stdbool.h>


#define MASTER 0
#define GROWTH_FACTOR 2
#define COMMUNICATION_SKIP 10
#define U 0
#define D 1
#define L 2
#define R 3
#define N 4


typedef struct walkData {
  int l_;
  int a_;
  int b_;
  int n_;
  int N_;
  double p_l_;
  double p_u_;
  double p_r_;
  double p_d_;
} walkData;


typedef struct Point {
  int x_;
  int y_;
  int step_;
} Point;


walkData* getInitialData(char** args) {
  walkData* walk_data = calloc(1, sizeof(walkData));
  assert(walk_data);
  walk_data->l_ = atoi(args[1]);
  walk_data->a_ = atoi(args[2]);
  walk_data->b_ = atoi(args[3]);
  walk_data->n_ = atoi(args[4]);
  walk_data->N_ = atoi(args[5]);
  walk_data->p_l_ = atof(args[6]);
  walk_data->p_r_ = atof(args[7]);
  walk_data->p_u_ = atof(args[8]);
  walk_data->p_d_ = atof(args[9]);
  return walk_data;
}


int getDirection(walkData* walk_data) {
  double p = (double)rand() / RAND_MAX;
  double p_u = walk_data->p_u_;
  double p_d = walk_data->p_d_;
  double p_l = walk_data->p_l_;
  double p_r = walk_data->p_r_;

  if (p_u <= p) {
    return U;
  }
  if (p_u + p_d <= p) {
    return D;
  }
  if (p_u + p_d + p_l <= p) {
    return L;
  }
  return R;
}


void randomStep(Point* points, int index, walkData* walk_data) {
  ++points[index].step_;
  int direction = getDirection(walk_data);
  switch (direction) {
    case U :
      ++points[index].y_;
      break;
    case D :
      --points[index].y_;
      break;
    case L :
      --points[index].x_;
      break;
    case R :
      ++points[index].x_;
      break;
  }
}


int transition(Point* points, int index, walkData* walk_data) {
  int l = walk_data->l_;
  if (points[index].y_ >= l) {
    points[index].y_ %= l;
    return U;
  }
  if (points[index].y_ < 0) {
    points[index].y_ += l;
    return D;
  }
  if (points[index].x_ < 0) {
    points[index].x_ += l;
    return L;
  }
  if (points[index].x_ >= l) {
    points[index].x_ %= l;
    return R;
  }
  return N;
}


int getRank(int rank, int direction, walkData* walk_data) {
  int a = walk_data->a_;
  int b = walk_data->b_;

  int row = rank / a;
  int column = rank % a;

  switch (direction) {
    case U :
      ++row;
      row %= b;
      break;
    case D :
      --row;
      row += b;
      row %= b;
      break;
    case L:
      --column;
      column += a;
      column %= a;
      break;
    case R:
      ++column;
      column %= a;
      break;
  }

  return (row * a + column);
}


void randomWalk(walkData* walk_data, int rank, int size) {
  double start_time = MPI_Wtime();
  
  int a = walk_data->a_;
  int b = walk_data->b_;
  int l = walk_data->l_;
  int i;

  int point_size = walk_data->N_;
  int capacity = GROWTH_FACTOR * point_size;
  Point* points = (Point*)calloc(capacity, sizeof(Point));
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
  
  for (i = 0; i < point_size; ++i) {
    points[i].x_ = rand() % l;
    points[i].y_ = rand() % l;
    points[i].step_ = 0;
  }

  int up_size = point_size;
  Point* up_points = (Point*)calloc(up_size, sizeof(Point));
  assert(up_points);
  int up_index;

  int down_size = point_size;
  Point* down_points = (Point*)calloc(down_size, sizeof(Point));
  assert(down_points);
  int down_index;

  int left_size = point_size;
  Point* left_points = (Point*)calloc(left_size, sizeof(Point));
  assert(left_points);
  int left_index;

  int right_size = point_size;
  Point* right_points = (Point*)calloc(right_size, sizeof(Point));
  assert(right_points);
  int right_index;

  int finished_points = 0;

  bool working = true;

  while (working) {

  up_index = 0;
  down_index = 0;
  left_index = 0;
  right_index = 0;

    for (i = 0; i < COMMUNICATION_SKIP; ++i) {

      int index = 0;

      while (index < point_size) {

        if (points[index].step_ < walk_data->n_) {
          randomStep(points, index, walk_data);
          int direction = transition(points, index, walk_data);

          switch (direction) {

            case U :
              if (up_index == up_size) {
                up_size *= GROWTH_FACTOR;
                up_points = (Point*)realloc(up_points, sizeof(Point) * up_size);
                assert(up_points);
              }
              up_points[up_index] = points[index];
              ++up_index;
              points[index] = points[point_size - 1];
              --point_size;
              --index;
              break;

            case D :
              if (down_index == down_size) {
                down_size *= GROWTH_FACTOR;
                down_points = (Point*)realloc(down_points, sizeof(Point) * down_size);
                assert(down_points);
              }
              down_points[down_index] = points[index];
              ++down_index;
              points[index] = points[point_size - 1];
              --point_size;
              --index;
              break;

            case L :
              if (left_index == left_size) {
                left_size *= GROWTH_FACTOR;
                left_points = (Point*)realloc(left_points, sizeof(Point) * left_size);
                assert(left_points);
              }
              left_points[left_index] = points[index];
              ++left_index;
              points[index] = points[point_size - 1];
              --point_size;
              --index;
              break;

            case R :
              if (right_index == right_size) {
                right_size *= GROWTH_FACTOR;
                right_points = (Point*)realloc(right_points, sizeof(Point) * right_size);
                assert(right_points);
              }
              right_points[right_index] = points[index];
              ++right_index;
              points[index] = points[point_size - 1];
              --point_size;
              --index;
              break;
          }

        } else {
          points[index] = points[point_size - 1];
          --point_size;
          --index;
          ++finished_points;
        }

        ++index;
      }
    }

    int* ranks = (int*)calloc(4, sizeof(int));

    ranks[0] = getRank(rank, U, walk_data);
    ranks[1] = getRank(rank, D, walk_data);
    ranks[2] = getRank(rank, L, walk_data);
    ranks[3] = getRank(rank, R, walk_data);

    int r_up_size = 0;
    int r_down_size = 0;
    int r_left_size = 0;
    int r_right_size = 0;

    MPI_Request* s_sizes = (MPI_Request*)calloc(4, sizeof(MPI_Request));
    assert(s_sizes);
    MPI_Request* r_sizes = (MPI_Request*)calloc(4, sizeof(MPI_Request));
    assert(r_sizes);

    MPI_Issend(&up_index, 1, MPI_INT, ranks[0], U, MPI_COMM_WORLD, r_sizes);
    MPI_Issend(&down_index, 1, MPI_INT, ranks[1], D, MPI_COMM_WORLD, r_sizes + 1);
    MPI_Issend(&left_index, 1, MPI_INT, ranks[2], L, MPI_COMM_WORLD, r_sizes + 2);
    MPI_Issend(&right_index, 1, MPI_INT, ranks[3], R, MPI_COMM_WORLD, r_sizes + 3);

    MPI_Irecv(&r_up_size, 1, MPI_INT, ranks[0], D, MPI_COMM_WORLD, s_sizes);
    MPI_Irecv(&r_down_size, 1, MPI_INT, ranks[1], U, MPI_COMM_WORLD, s_sizes + 1);
    MPI_Irecv(&r_left_size, 1, MPI_INT, ranks[2], R, MPI_COMM_WORLD, s_sizes + 2);
    MPI_Irecv(&r_right_size, 1, MPI_INT, ranks[3], L, MPI_COMM_WORLD, s_sizes + 3);

    MPI_Status* s_status = (MPI_Status*)calloc(4, sizeof(MPI_Status));
    assert(s_status);
    MPI_Status* r_status = (MPI_Status*)calloc(4, sizeof(MPI_Status));
    assert(r_status);

    for (i = 0; i < 4; ++i) {
      MPI_Wait(r_sizes + i, s_status + i);
      MPI_Wait(s_sizes + i, r_status + i);
    }

    Point* r_up_points = calloc(r_up_size, sizeof(Point));
    assert(r_up_points);
    Point* r_down_points = calloc(r_down_size, sizeof(Point));
    assert(r_down_points);
    Point* r_left_points = calloc(r_left_size, sizeof(Point));
    assert(r_left_points);
    Point* r_right_points = calloc(r_right_size, sizeof(Point));
    assert(r_right_points);

    MPI_Isend(up_points, up_index * sizeof(Point), MPI_BYTE, ranks[0], U, MPI_COMM_WORLD, r_sizes);
    MPI_Isend(down_points, down_index * sizeof(Point), MPI_BYTE, ranks[1], D, MPI_COMM_WORLD, r_sizes + 1);
    MPI_Isend(left_points, left_index * sizeof(Point), MPI_BYTE, ranks[2], L, MPI_COMM_WORLD, r_sizes + 2);
    MPI_Isend(right_points, right_index * sizeof(Point),MPI_BYTE, ranks[3], R, MPI_COMM_WORLD, r_sizes + 3);

    MPI_Irecv(r_up_points, r_up_size * sizeof(Point), MPI_BYTE, ranks[0], D, MPI_COMM_WORLD, s_sizes);
    MPI_Irecv(r_down_points, r_down_size * sizeof(Point), MPI_BYTE, ranks[1], U, MPI_COMM_WORLD, s_sizes + 1);
    MPI_Irecv(r_left_points, r_left_size * sizeof(Point), MPI_BYTE, ranks[2], R, MPI_COMM_WORLD, s_sizes + 2);
    MPI_Irecv(r_right_points, r_right_size * sizeof(Point), MPI_BYTE, ranks[3], L, MPI_COMM_WORLD, s_sizes + 3);

    for (i = 0; i < 4; ++i) {
      MPI_Wait(r_sizes + i, s_status + i);
      MPI_Wait(s_sizes + i, r_status + i);
    }

    for (i = 0; i < r_up_size; ++i) {
      if (point_size == capacity) {
        capacity *= GROWTH_FACTOR;
        points = (Point*)realloc(points, sizeof(Point) * capacity);
        assert(points);
      }
      points[point_size] = r_up_points[i];
      ++point_size;
    }

    for (i = 0; i < r_down_size; ++i) {
      if (point_size == capacity) {
        capacity *= GROWTH_FACTOR;
        points = (Point*)realloc(points, sizeof(Point) * capacity);
        assert(points);
      }
      points[point_size] = r_down_points[i];
      ++point_size;
    }

    for (i = 0; i < r_left_size; ++i) {
      if (point_size == capacity) {
        capacity *= GROWTH_FACTOR;
        points = (Point*)realloc(points, sizeof(Point) * capacity);
        assert(points);
      }
      points[point_size] = r_left_points[i];
      ++point_size;
    }

    for (i = 0; i < r_right_size; ++i) {
      if (point_size == capacity) {
        capacity *= GROWTH_FACTOR;
        points = (Point*)realloc(points, sizeof(Point) * capacity);
        assert(points);
      }
      points[point_size] = r_right_points[i];
      ++point_size;
    }      

    int sum_finished_points = 0;
    MPI_Reduce(&finished_points, &sum_finished_points, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Request* send = (MPI_Request*)calloc(size, sizeof(MPI_Request));
    MPI_Request* receive = (MPI_Request*)calloc(size, sizeof(MPI_Request));

    if (rank == MASTER) {
      if (sum_finished_points == size * walk_data->N_) {
        working = false;
      }
      for (i = 0; i < size; ++i) {
        MPI_Isend(&working, 1 ,MPI_INT, i, 0, MPI_COMM_WORLD, send + i);
      }
    }

    MPI_Irecv(&working, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, receive + rank);

    MPI_Barrier(MPI_COMM_WORLD);

    free(r_up_points);
    free(r_down_points);
    free(r_left_points);
    free(r_right_points);
    free(s_sizes);
    free(r_sizes);
    free(s_status);
    free(r_status);
    free(send);
    free(receive);
  }

  double end_time = MPI_Wtime();

  int* result = (int*)calloc(2 * size, sizeof(int));
  assert(result);
  
  int* pair = (int*)calloc(2, sizeof(int));
  assert(pair);

  pair[0] = rank;
  pair[1] = finished_points;

  MPI_Gather(pair, 2, MPI_INT, result, 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == MASTER) {
    FILE* f = fopen("stats.txt", "w");
    fprintf(f, "%d %d %d %d %d %f %f %f %f", walk_data->l_, walk_data->a_, walk_data->b_, walk_data->n_, walk_data->N_, walk_data->p_l_, walk_data->p_r_, walk_data->p_u_, walk_data->p_d_);
    fprintf(f, " %1.3lfs\n", end_time - start_time);
    for (i = 0; i < 2 * size; i += 2) {
      fprintf(f, "%d : %d\n", result[i], result[i + 1]);
    }
    fclose(f);
  }

  free(seeds);
  free(points);
  free(up_points);
  free(down_points);
  free(left_points);
  free(right_points);
  free(result);
  free(pair);
}


int main(int argc, char **argv) {
  walkData* walk_data = getInitialData(argv);

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assert(size == walk_data->a_ * walk_data->b_);
  randomWalk(walk_data, rank, size);

  MPI_Finalize();

  free(walk_data);
  return 0;
}

