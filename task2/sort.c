#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>


typedef struct Data {
  int n_;
  int m_;
  int P_;
  double time_;
  int* initial_data_;
  int* sorted_data_;
} Data;


Data* initializeData(char** argv) {
  Data* data = calloc(1, sizeof(Data));
  if (data == NULL) {
    fprintf(stderr, "Failed at allocating memory in initializeData function");
    free(data);
    exit(0);
  }
  data->n_ = atoi(argv[1]);
  data->m_ = atoi(argv[2]);
  data->P_ = atoi(argv[3]);
  data->time_ = 0;

  data->initial_data_ = calloc(data->n_, sizeof(int));
  if (data->initial_data_ == NULL) {
    fprintf(stderr, "Failed at allocating memory in initializeData function");
    free(data->initial_data_);
    free(data);
    exit(0);
  }

  data->sorted_data_ = calloc(data->n_, sizeof(int));
  if (data->sorted_data_ == NULL) {
    fprintf(stderr, "Failed at allocating memory in initializeData function");
    free(data->initial_data_);
    free(data->sorted_data_);
    free(data);
    exit(0);
  }
  
  srand(time(NULL));
  for (int i = 0; i < data->n_; ++i) {
    data->initial_data_[i] = rand();
    data->sorted_data_[i] = data->initial_data_[i];
  }
  return data;
}


void printToFile(FILE* file, Data* data) {
  for (int i = 0; i < data->n_; ++i) {
    fprintf(file, "%d ", data->initial_data_[i]);
  }
  fprintf(file, "\n");
  for (int i = 0; i < data->n_; ++i) {
    fprintf(file, "%d ", data->sorted_data_[i]);
  }
}


int cmpfunc(const void* el1, const void* el2) {
  return (*(int*)el1 - *(int*)el2);
}


// (left; right]
int leftBinSearch(int* array, int start_position, int end_position, int element) {
  int left = start_position - 1;
  int right = end_position;
  while (right - left > 1) {
    int middle = (left + right) / 2;
    if (element > array[middle]) {
      left = middle;
    } else {
      right = middle;
    }
  }
  return right;
}


// [left; right)
int* mergedChunk(int* array, int first_left, int first_right, int second_left, int second_right, int* size) {
  *size = first_right - first_left + second_right - second_left;
  int* tmp = (int*)calloc(*size, sizeof(int));
  if (tmp == NULL) {
    fprintf(stderr, "Failed at memory allocating in mergedChunk function");
    free(tmp);
    exit(0);
  }
  int first_ind = 0;
  int second_ind = 0;
  while (first_left + first_ind < first_right && second_left + second_ind < second_right) {
    if (array[first_left + first_ind] < array[second_left + second_ind]) {
      tmp[first_ind + second_ind] = array[first_left + first_ind];
      first_ind += 1;
    } else {
      tmp[first_ind + second_ind] = array[second_left + second_ind];
      second_ind += 1;
    }
  }

  while (first_left + first_ind < first_right) {
    tmp[first_ind + second_ind] = array[first_left + first_ind];
    first_ind += 1;
  }

  while (second_left + second_ind < second_right) {
    tmp[first_ind + second_ind] = array[second_left + second_ind];
    second_ind += 1;
  }
  return tmp;
}


void placeBack(int* array, int start_position, int* tmp, int size) {
  for (int i = 0; i < size; ++i) {
    array[start_position + i] = tmp[i];
  }
}


// [left; right)
void parallel_sort(int* array, int left, int right, int chunk_size) {
  if (right - left <= chunk_size) {
    qsort(array + left, right - left, sizeof(int), cmpfunc);
    return;
  }
  int first_size;
  int* first_chunk;
  int second_size;
  int* second_chunk;
  int middle = (right - left) / 2 + left;
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task
      parallel_sort(array, left, middle, chunk_size);
      #pragma omp task
      parallel_sort(array, middle, right, chunk_size);
    }
    int median = (middle - left) / 2 + left;
    int index = leftBinSearch(array, middle, right, array[median]);
    #pragma omp single
    {
      #pragma omp task
      first_chunk = mergedChunk(array, left, median, middle, index, &first_size);
      #pragma omp task
      second_chunk = mergedChunk(array, median, middle, index, right, &second_size);
    }
    #pragma omp single
    {
      #pragma omp task
      placeBack(array, left, first_chunk, first_size);
      #pragma omp task
      placeBack(array, left + first_size, second_chunk, second_size);
    }
  }
  free(first_chunk);
  free(second_chunk);
}


bool sorted(Data* data) {
  for (int i = 0; i < data->n_ - 1; ++i) {
    if (data->sorted_data_[i] > data->sorted_data_[i + 1]) {
      return false;
    }
  }
  return true;
}


bool isSame(Data* data) {
  int xor = 0;
  for (int i = 0; i < data->n_; ++i) {
    xor ^= data->initial_data_[i];
  }
  for (int i = 0; i < data->n_; ++i) {
    xor ^= data->sorted_data_[i];
  }
  if (xor == 0) {
    return true;
  } else {
    return false;
  }
}



int main(int argc, char** argv) {
  FILE* data_txt = fopen("data.txt", "w");
  FILE* stats_txt = fopen("stats.txt", "w");
  Data* data = initializeData(argv);
  omp_set_num_threads(data->P_);

  data->time_ = omp_get_wtime();
  parallel_sort(data->sorted_data_, 0, data->n_, data->m_);
  data->time_ = omp_get_wtime() - data->time_;

  printToFile(data_txt, data);
  bool flag_sorted = sorted(data);
  bool flag_same = isSame(data);
  if (flag_sorted && flag_same) {
    fprintf(stats_txt, "%2.5fs\n", data->time_);
  } else {
    if (!flag_sorted) {
      fprintf(stats_txt, "Initial data wasn't sorted");
    } else {
      fprintf(stats_txt, "Sorted data has been changed");
    }
  }

  fclose(data_txt);
  fclose(stats_txt);
  free(data->initial_data_);
  free(data->sorted_data_);
  free(data);

  return 0;
}
