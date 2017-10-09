#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
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

typedef struct Args {
  Data* data_;
  int ind_;
  int cur_step_;
} Args;


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


int min(int a, int b) {
  if (a < b) {
    return a;
  } else {
    return b;
  }
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


void* chunkSort(void* param) {
  Args* args = param;
  Data* data = args->data_;
  int ind = args->ind_;
  int cur_step = args->cur_step_;
  int left = cur_step * ind;
  int right = min(cur_step * (ind + 1), data->n_);
  while(left < data->n_) {
    qsort(data->sorted_data_ + left, right - left, sizeof(int), cmpfunc);

    left += cur_step * data->P_;
    right += cur_step * data->P_;
    right = min(right, data->n_);
  }
  return NULL;
}


void* mergeSort(void* param) {
  Args* args = param;
  Data* data = args->data_;
  int cur_step = args->cur_step_;
  int limit = data->n_;
  int chunk_num = (limit + cur_step - 1) / cur_step;
  if (chunk_num % 2 != 0) {
    limit = (chunk_num - 1) * cur_step;
  }
  int left, right, middle, median;
  int first_size, second_size;
  for (int i = 2 * args->ind_ * cur_step; i < limit; i += 2 * data->P_ * cur_step) {
    int left = i;
    right = left + 2 * cur_step;
    if (right < data->n_) {
      middle = (right - left) / 2 + left;
    } else {
      right = data->n_;
      middle = left + cur_step;
    }
    median = (middle - left) / 2 + left;
    int index = leftBinSearch(data->sorted_data_, middle, right, data->sorted_data_[median]);
    int* first_chunk = mergedChunk(data->sorted_data_, left, median, middle, index, &first_size);
    int* second_chunk = mergedChunk(data->sorted_data_, median, middle, index, right, &second_size);
    placeBack(data->sorted_data_, left, first_chunk, first_size);
    placeBack(data->sorted_data_, left + first_size, second_chunk, second_size);
    free(first_chunk);
    free(second_chunk);
  }
  if (chunk_num % 2 != 0 && args->ind_ == ((chunk_num - 1) / 2 + data->P_ - 1) % data->P_) {
    int left = (chunk_num - 3) * cur_step;
    int right = left + cur_step * 2;
    int middle = (right - left) / 2 + left;
    int border = data->n_;
    int index = leftBinSearch(data->sorted_data_, right, border, data->sorted_data_[middle]);
    int first_size;
    int second_size;
    int* first_chunk = mergedChunk(data->sorted_data_, left, middle, right, index, &first_size);
    int* second_chunk = mergedChunk(data->sorted_data_, middle, right, index, border, &second_size);
    placeBack(data->sorted_data_, left, first_chunk, first_size);
    placeBack(data->sorted_data_, left + first_size, second_chunk, second_size);
    free(first_chunk);
    free(second_chunk);
  }
  return NULL;
}


void sort(Data* data) {
  pthread_t threads[data->P_];
  Args args[data->P_];
  for (int i = 0; i < data->P_; ++i) {
    args[i].data_ = data;
    args[i].ind_ = i;
    args[i].cur_step_ = data->m_;
  }

  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  for (int i = 0; i < data->P_; ++i) {
    pthread_create(&threads[i], NULL, &chunkSort, (void*)&args[i]);
  }

  for (int i = 0; i < data->P_; ++i) {
    pthread_join(threads[i], NULL);
  }
  gettimeofday(&tv2, NULL);
  data->time_ += ((double)tv2.tv_sec + (double)tv2.tv_usec / 1000000.0) - ((double)tv1.tv_sec + (double)tv1.tv_usec / 1000000.0);
 
  int chunk_step = data->m_;
  for (int step = data->m_; step < data->n_; step *= 2) {
    for (int i = 0; i < data->P_; ++i) {
      args[i].cur_step_ = step;
    }
    gettimeofday(&tv1, NULL);
    for (int i = 0; i < data->P_; ++i) {
      pthread_create(&threads[i], NULL, &mergeSort, (void*)&args[i]);
    }
    for (int i = 0; i < data->P_; ++i) {
      pthread_join(threads[i], NULL);
    }
    gettimeofday(&tv2, NULL);
    data->time_ += ((double)tv2.tv_sec + (double)tv2.tv_usec / 1000000.0) - ((double)tv1.tv_sec + (double)tv1.tv_usec / 1000000.0);
  }
}


bool isSorted(Data* data) {
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
  sort(data);

  printToFile(data_txt, data);
  bool flag_sorted = isSorted(data);
  bool flag_same = isSame(data);

  if (flag_sorted && flag_same) {
    fprintf(stats_txt, "%2.5fs %d %d %d\n", data->time_, data->n_, data->m_, data->P_);
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
