// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Brent-kung parallel scan with double buffering: nlog(n)
__global__ void bk_scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // double buffer
  __shared__ float T[2*BLOCK_SIZE];
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  int start_idx = 2 * bi * blockDim.x;
  // each thread need two load two elements into shared memory
  if (ti + start_idx < len) 
    T[ti] = input[start_idx + ti];
  if (ti + start_idx + blockDim.x < len) 
    T[ti + blockDim.x] = input[start_idx + blockDim.x + ti];

  // reduction step 
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
    __syncthreads();
    int idx = (ti + 1) * stride * 2 - 1;
    if (idx < 2 * BLOCK_SIZE && (idx - stride) >= 0)
      T[idx] += T[idx-stride];
    stride *= 2;
  }

  // post scan step
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    __syncthreads();
    int idx = (ti + 1) * stride * 2 - 1;
    if ((idx + stride) < 2 * BLOCK_SIZE)
      T[idx + stride] += T[idx];
    stride /= 2;
  }

  if (ti + start_idx < len)
    output[ti + start_idx] = T[ti];
  if (ti + start_idx + blockDim.x < len)
    output[ti + start_idx + blockDim.x] = T[ti + blockDim.x];
}

__global__ void add(float *block_sums, float *input, int len) {
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  int start_idx = 2 * (bi + 1) * blockDim.x;
  int index = start_idx + ti;
  if (index < len) {
    input[index] += block_sums[bi];
  }
  if (index + blockDim.x < len)
    input[index + blockDim.x] += block_sums[bi];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  unsigned int grid_size = (numElements-1) / (2 * BLOCK_SIZE) + 1;
  dim3 gridDim{grid_size, 1, 1};
  dim3 blockDim{BLOCK_SIZE, 1, 1};
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  // step 1 scan block sum
  bk_scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);
  // step 2 scan block sums
  float * devBlockSumsInput;
  float * hostBlockSumsInput;
  float * devBlockSumsOutput;
  wbCheck(cudaMalloc((void **)&devBlockSumsInput, grid_size * sizeof(float)));
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  hostBlockSumsInput = (float *)malloc(grid_size * sizeof(float));
  for (int i = 0; i < grid_size; i++) 
    hostBlockSumsInput[i] = hostOutput[(i+1)*2*BLOCK_SIZE-1];
  wbCheck(cudaMemcpy(devBlockSumsInput, hostBlockSumsInput, grid_size * sizeof(float), \
        cudaMemcpyHostToDevice));
  wbCheck(cudaMalloc((void **)&devBlockSumsOutput, grid_size * sizeof(float)));
  bk_scan<<<1, (grid_size-1)/2 + 1>>>(devBlockSumsInput, devBlockSumsOutput, grid_size);
  // step 3 add scanned block sum i to all values in block i + 1
  add<<<gridDim, blockDim>>>(devBlockSumsOutput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
