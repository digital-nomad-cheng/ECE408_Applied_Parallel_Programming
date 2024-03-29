#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

int ceil(int a, int b) {
  return (a - 1 + b) / b;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//@@ Define any useful program-wide constants here
#define TILE_SIZE 8
#define MASK_WIDTH 3
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

// global memory version
__global__ void conv3d_navie(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int radius = MASK_WIDTH/2;
  int x_start = x - radius;
  int y_start = y - radius;
  int z_start = z - radius;
  float result = 0.0f;
  for (int i = 0; i < MASK_WIDTH; i++) {
    for (int j = 0; j < MASK_WIDTH; j++) {
      for (int k = 0; k < MASK_WIDTH; k++) {
        int zz = z_start + i; 
        int yy = y_start + j;
        int xx = x_start + k;
        if ((0 <= xx) && (xx < x_size) && \
            (0 <= yy) && (yy < y_size) && \
            (0 <= zz) && (zz < z_size)) {
            result += input[xx +  yy * x_size + zz * x_size * y_size] * deviceKernel[i][j][k];
        }
      }
    }
  }
        
  if ((0 <= x) && (x < x_size) && \
      (0 <= y) && (y < y_size) && \
      (0 <= z) && (z < z_size))
    output[x+y*x_size+z*x_size*y_size] = result;
}

// shared memory version
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N_dev[TILE_SIZE][TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  // global index 
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int index = z * y_size * x_size + y * x_size + x;

  if ((0 <= x) && (x < x_size) && \
      (0 <= y) && (y < y_size) && \
      (0 <= z) && (z < z_size)) {
    N_dev[tz][ty][tx] = input[index];
  } else {
    return;
  }

  __syncthreads();
  
  int radius = MASK_WIDTH / 2;
  // valiad shared memory access region
  int current_tile_x = blockIdx.x * blockDim.x;
  int current_tile_y = blockIdx.y * blockDim.y;
  int current_tile_z = blockIdx.z * blockDim.z;
  int next_tile_x = (blockIdx.x + 1) * blockDim.x;
  int next_tile_y = (blockIdx.y + 1) * blockDim.y;
  int next_tile_z = (blockIdx.z + 1) * blockDim.z;
  // start index for inner product
  int x_start = x - radius;
  int y_start = y - radius;
  int z_start = z - radius;

  float result = 0.0f;
  for (int i = 0; i < MASK_WIDTH; i++) {
    for (int j = 0; j < MASK_WIDTH; j++) {
      for (int k = 0; k < MASK_WIDTH; k++) {
        int zz = z_start + i; 
        int yy = y_start + j;
        int xx = x_start + k;
        if ((0 <= xx) && (xx < x_size) && \
            (0 <= yy) && (yy < y_size) && \
            (0 <= zz) && (zz < z_size)) {
          if ((current_tile_x <= xx) && (xx < next_tile_x) && \
              (current_tile_y <= yy) && (yy < next_tile_y) && \
              (current_tile_z <= zz) && (zz < next_tile_z))
            result += N_dev[tz-radius+i][ty-radius+j][tx-radius+k] * deviceKernel[i][j][k];
          else
            result += input[xx +  yy * x_size + zz * x_size * y_size] * deviceKernel[i][j][k];
        }
      }
    }
  }
  if ((0 <= x) && (x < x_size) && \
      (0 <= y) && (y < y_size) && \
      (0 <= z) && (z < z_size))  
    output[index] = result;
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  gpuErrchk(cudaMalloc((void **)&deviceInput, (inputLength - 3)*sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&deviceOutput, (inputLength - 3)*sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  gpuErrchk(cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice)); 
  gpuErrchk(cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float)));
  cudaDeviceSynchronize();
  wbTime_stop(Copy, "Copying data to the GPU");
  
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 gridDim{ceil(x_size, TILE_SIZE), ceil(y_size, TILE_SIZE), ceil(z_size, TILE_SIZE)};
  dim3 blockDim{TILE_SIZE, TILE_SIZE, TILE_SIZE};
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3)*sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
