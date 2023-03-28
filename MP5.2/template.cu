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

// Kogge-stone parallel scan with double buffering: nlog(n)
__global__ void ks_scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // double buffer
  __shared__ float T0[BLOCK_SIZE];
  __shared__ float T1[BLOCK_SIZE];
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  int index = bi * blockDim.x + ti;
  float * src = T0;
  float * dest = T1;
  if (index < len) {
    T0[ti] = input[index];
    T1[ti] = T0[ti];
  }

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (ti >= stride)
      dest[ti] = src[ti] + src[ti - stride];
    else
      dest[ti] = src[ti];
    float * temp = src;
    src = dest;
    dest = temp;
  }
  if (index < len)
    output[index] = src[ti];
}
__global__ void add(float *block_sums, float *input, int len) {
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  int index = (bi + 1) * blockDim.x + ti;
  if (index < len) {
    input[index] += block_sums[bi];
  }
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
  unsigned int grid_size = (numElements-1) / BLOCK_SIZE + 1;
  dim3 gridDim{grid_size, 1, 1};
  dim3 blockDim{BLOCK_SIZE, 1, 1};
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  // step 1 scan block sum
  ks_scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);
  // step 2 scan block sums
  float * devBlockSumsInput;
  float * hostBlockSumsInput;
  float * devBlockSumsOutput;
  wbCheck(cudaMalloc((void **)&devBlockSumsInput, grid_size * sizeof(float)));
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
  hostBlockSumsInput = (float *)malloc(grid_size * sizeof(float));
  for (int i = 0; i < grid_size; i++) 
    hostBlockSumsInput[i] = hostOutput[(i+1)*BLOCK_SIZE-1];
  wbCheck(cudaMemcpy(devBlockSumsInput, hostBlockSumsInput, grid_size * sizeof(float), \
        cudaMemcpyHostToDevice));
  wbCheck(cudaMalloc((void **)&devBlockSumsOutput, grid_size * sizeof(float)));
  ks_scan<<<1, grid_size>>>(devBlockSumsInput, devBlockSumsOutput, grid_size);
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
