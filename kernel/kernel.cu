#include <stdio.h>

#define INPUT_DIM 100
#define FILTER_DIM 5
#define CONV_LAYER_SIZE 10
#define REGION_DIM 20
#define CONV_OUT_DIM (INPUT_DIM / FILTER_DIM)
#define OUT_LAYER_SIZE 10
#define OUT_NEURON_DIM (CONV_OUT_DIM * CONV_OUT_DIM * CONV_LAYER_SIZE)

extern "C" __global__ void convolution(const double input[INPUT_DIM][INPUT_DIM], const double conv_layer[CONV_LAYER_SIZE][FILTER_DIM][FILTER_DIM], double conv_output[CONV_LAYER_SIZE][CONV_OUT_DIM][CONV_OUT_DIM]) {
  int i = blockIdx.x * FILTER_DIM; 
  int j = blockIdx.y * FILTER_DIM;
  int z = threadIdx.x; 

  if (i + FILTER_DIM > INPUT_DIM || j + FILTER_DIM > INPUT_DIM || z >= CONV_LAYER_SIZE) {
    return;
  }

  double prod = 0;
  for (int x = 0; x < FILTER_DIM; x++) {
    for (int y = 0; y < FILTER_DIM; y++) {
      prod += (input[i + x][j + y] * conv_layer[z][x][y]);
    }
  }
  conv_output[z][i / FILTER_DIM][j / FILTER_DIM] = prod;
}

extern "C" __global__ void relu(double conv_output[CONV_LAYER_SIZE][CONV_OUT_DIM][CONV_OUT_DIM]) {
  int i = blockIdx.x; 
  int j = blockIdx.y;

  int z = threadIdx.x; 

  if (i >= CONV_OUT_DIM || j >= CONV_OUT_DIM || z >= CONV_LAYER_SIZE) {
    return;
  }

  if (conv_output[z][i][j] < 0) {
    conv_output[z][i][j] = 0.0;
  }
}

extern "C" __global__ void output(double conv_output[CONV_LAYER_SIZE][CONV_OUT_DIM][CONV_OUT_DIM], double weights[OUT_LAYER_SIZE][OUT_NEURON_DIM], double output[OUT_LAYER_SIZE]) {
  int z = threadIdx.x; 

  if (x >= CONV_OUT_DIM || y >= CONV_OUT_DIM || z >= OUT_LAYER_SIZE) {
    return;
  }

  double prod = 0;
  for (int i = 0; i < CONV_OUT_DIM; i++) {
    for (int j = 0; j < CONV_OUT_DIM; j++) {
      for (int k = 0; k < OUT_LAYER_SIZE; k++) {
        int id = k * 400 + i * 20 + j;
        prod += (conv_output[k][i][j] * weights[z][id]);
      }
    }
  }

  output[z] = prod;
}
