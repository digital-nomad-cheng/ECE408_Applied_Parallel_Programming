// Histogram Equalization

#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}



#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16 // for easier histogram cdf scan computation

__constant__ float const_cdf[HISTOGRAM_LENGTH];

//@@ insert code here
/*
 * float image to unsigned char image
 */
__global__ void float2uchar(unsigned char *output, const float *input, int width, int height, int channels) 
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    output[channels*idx + 0] = \
          (unsigned char) (255.0f * input[channels*idx + 0]);
    output[channels*idx + 1] = \
          (unsigned char) (255.0f * input[channels*idx + 1]);
    output[channels*idx + 2] = \
          (unsigned char) (255.0f * input[channels*idx + 2]);
  }
}

/*
 * rgb image to grayscale image
 */
__global__ void rgb2gray(unsigned char *gray, const unsigned char *rgb, int width, int height, int channels)
{
  int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
  int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
  if (tid_x < width && tid_y < height) {
    int idx = tid_y * width + tid_x;
    unsigned char r = rgb[3*idx];
    unsigned char g = rgb[3*idx+1];
    unsigned char b = rgb[3*idx+2];
    gray[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    // For debugging rgb to gray
    // gray[idx*3 + 0] =  (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    // gray[idx*3 + 1] =  (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    // gray[idx*3 + 2] =  (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

/*
 * Compute histogram of gray image
 */
__global__ void histogram(int *hists, const unsigned char *gray, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if ( x < width && y < height) {
    int idx = y * width + x;
    atomicAdd(&(hists[gray[idx]]), 1);
  }
}

/*
 * Compute cdf using histogram - a inherent inclusive prefix sum problem
 */
__global__ void cdf(float *output, const int *input, int width, int height)
{
  float size = (float) width * height;
  __shared__ float T[HISTOGRAM_LENGTH];
  
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  int start_idx = 2 * bi * blockDim.x;
  // each thread need two load two elements into shared memory
  if (ti + start_idx < HISTOGRAM_LENGTH) 
    T[ti] = input[start_idx + ti] / size;
  if (ti + start_idx + blockDim.x < HISTOGRAM_LENGTH) 
    T[ti + blockDim.x] = input[start_idx + blockDim.x + ti] / size;

  // reduction step 
  int stride = 1;
  while (stride < HISTOGRAM_LENGTH) {
    __syncthreads();
    int idx = (ti + 1) * stride * 2 - 1;
    if (idx < HISTOGRAM_LENGTH && (idx - stride) >= 0)
      T[idx] += T[idx-stride];
    stride *= 2;
  }

  // post scan step
  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    int idx = (ti + 1) * stride * 2 - 1;
    if ((idx + stride) < HISTOGRAM_LENGTH)
      T[idx + stride] += T[idx];
    stride /= 2;
  }

  if (ti + start_idx < HISTOGRAM_LENGTH)
    output[ti + start_idx] = T[ti] + input[ti] / size;
  if (ti + start_idx + blockDim.x < HISTOGRAM_LENGTH)
    output[ti + start_idx + blockDim.x] = T[ti + blockDim.x] + input[ti + blockDim.x] / size;
}

/*
 * perform histogram equalization
 */
// def correct_color(val)
// 	return clamp(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0, 255.0)
// end

// def clamp(x, start, end)
// 	return min(max(x, start), end)
// end
// __device__ unsigned char 
__device__ unsigned char correct_color(int val)
{
  int tmp = (int) 255.0f * (const_cdf[val] - const_cdf[0]) / (1.0f - const_cdf[0]);
  return (unsigned char) min(max(0, tmp), 255);
}
// }
__global__ void equalization(unsigned char *output, const unsigned char *input, int width, int height, int channels)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    for (int c = 0; c < channels; c++) {
      int val = (int)input[idx*channels + c];
      output[idx*channels + c] = correct_color(val);
    }
  }
}

/*
 * unsigned char image to float image
 */
__global__ void uchar2float(float *output, const unsigned char *input, int width, int height, int channels) 
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    output[channels*idx + 0] = \
            (float) (input[channels*idx + 0] / 255.0);
    output[channels*idx + 1] = \
            (float) (input[channels*idx + 1] / 255.0);
    output[channels*idx + 2] = \
            (float) (input[channels*idx + 2] / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  // outputImage = wbImage_new(imageHeight, imageWidth, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  std::cout << "image width: " << imageWidth << " image height: " << imageHeight << \
    " image channels: " << imageChannels << std::endl;
  
  
  //@@ insert code here
  //@ 1. allocate memory on device
  float *dev_float;
  unsigned char *dev_uchar;
  unsigned char *dev_gray;
  int *dev_hists;
  float *dev_cdf;
  unsigned char *dev_eq;
  // float *dev_gray_float;
  wbCheck(cudaMalloc((void **)&dev_float, (imageWidth*imageHeight*imageChannels)*sizeof(float)));
  wbCheck(cudaMalloc((void **)&dev_uchar, (imageWidth*imageHeight*imageChannels)*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&dev_gray, (imageWidth*imageHeight)*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&dev_hists, HISTOGRAM_LENGTH * sizeof(int)));
  wbCheck(cudaMemset(dev_hists, 0, HISTOGRAM_LENGTH * sizeof(int)));
  wbCheck(cudaMalloc((void **)&dev_cdf, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&dev_eq, (imageWidth*imageHeight*imageChannels)*sizeof(unsigned char)));

  // cudaMalloc((void **)&dev_gray_float, (imageWidth*imageHeight*imageChannels)*sizeof(float));
  
  wbCheck(cudaMemcpy(dev_float, hostInputImageData, (imageWidth*imageHeight*imageChannels)*sizeof(float), \
      cudaMemcpyHostToDevice));
  //@ 2. define kernel launch parameters
  dim3 grid_dim{(imageWidth - 1) / BLOCK_SIZE + 1, (imageHeight - 1) / BLOCK_SIZE + 1, 1};
  dim3 block_dim{BLOCK_SIZE, BLOCK_SIZE, 1};
  
  // algo step 1: float to uchar
  float2uchar<<<grid_dim, block_dim>>>(dev_uchar, dev_float, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("Launching float to uchar kernel failed!");

  // algo step 2: rgb to gray
  rgb2gray<<<grid_dim, block_dim>>>(dev_gray, dev_uchar, imageWidth, imageHeight,  imageChannels);
  checkCUDAErrorWithLine("Launching rgb to gray kernel failed!");
  
  /*
   * Debugging rgb 2 gray
   */
  
  /*
  unsigned char *host_gray;
  host_gray = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));
  cudaMemcpy(host_gray,dev_gray, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  for (int y = 0; y < 9; y++) {
    std::cout << std::endl;
    for (int x = 0; x < 9; x++) {
      std::cout << (int)host_gray[y * imageWidth + x] << " ";
    }
  }
  */

  // algo step 3: compute histogram using gray image
  histogram<<<grid_dim, block_dim>>>(dev_hists, dev_gray, imageWidth, imageHeight); 
  checkCUDAErrorWithLine("Launching histogram kernel failed!");
  /*
   * Debugging histogram
   */ 
  
  /*
  int *host_hists;
  host_hists = (int *)malloc(HISTOGRAM_LENGTH * sizeof(int));
  cudaMemcpy(host_hists, dev_hists, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < HISTOGRAM_LENGTH; i++)
    std::cout << "hist[" << i << "]: " << host_hists[i] << std::endl;
  */

  // algo step 4: compute cfg using histogram
  cdf<<<1, HISTOGRAM_LENGTH/2>>>(dev_cdf, dev_hists, imageWidth, imageHeight);
  checkCUDAErrorWithLine("Launching cdf kernel failed!");
  /*
   * Debugging cdf
   */
  
  // float *host_cdf;
  // host_cdf = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  // cudaMemcpy(host_cdf, dev_cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < HISTOGRAM_LENGTH; i++)
  //   std::cout << "cdf[" << i << "]: " << host_cdf[i] << std::endl;
  // assert(host_cdf[HISTOGRAM_LENGTH-1] == 1), "Wrong cdf calculation.";
  

  // algo step 5: histogram equalization
  // copy the calcuated cdf to constant memory
  wbCheck(cudaMemcpyToSymbol(const_cdf, dev_cdf, HISTOGRAM_LENGTH * sizeof(float)));
  equalization<<<grid_dim, block_dim>>>(dev_eq, dev_uchar, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("Launching equalization kernel failed!");

  // algo step 6: uchar to float
  uchar2float<<<grid_dim, block_dim>>>(dev_float, dev_eq, imageWidth, imageHeight, imageChannels);
  checkCUDAErrorWithLine("Launching uchar 2 float kernel failed!");

  cudaMemcpy(hostOutputImageData, dev_float, (imageWidth*imageHeight*imageChannels)*sizeof(float), \
      cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();

  wbSolution(args, outputImage);


  //@@ insert code here
  // memory cleanup
  cudaFree(dev_float);
  cudaFree(dev_uchar);
  cudaFree(dev_gray);
  cudaFree(dev_hists);
  cudaFree(dev_cdf);
  cudaFree(dev_eq);
  
  wbImage_delete(inputImage);
  wbImage_delete(outputImage);

  return 0;
}
