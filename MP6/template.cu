// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

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
    // gray[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    // For debugging rgb to gray
    gray[idx*3 + 0] =  (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    gray[idx*3 + 1] =  (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    gray[idx*3 + 2] =  (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

/*
 * compute histogram of gray image
 */
// __global__ void histogram(unsigned char *output, const unsigned char *gray, int w, int h)
// {
// }

/*
 * perform histogram equalization
 */


/*
 * unsigned char image to float image
 */
__global__ void uchar2float(float *output, const unsigned char *input, int width, int height, int channels) 
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    // output[channels*idx + 0] = \
    //         (float) (input[channels*idx + 0] / 255.0);
    // output[channels*idx + 1] = \
    //         (float) (input[channels*idx + 1] / 255.0);
    // output[channels*idx + 2] = \
    //         (float) (input[channels*idx + 2] / 255.0);
    for (int c = 0; c < channels; c++) {
      output[channels*idx + c] = \
            (float) (input[channels*idx + c] / 255.0);
    }
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
  unsigned char *dev_gray_uchar;
  float *dev_gray_float;
  cudaMalloc((void **)&dev_float, (imageWidth*imageHeight*imageChannels)*sizeof(float));
  cudaMalloc((void **)&dev_uchar, (imageWidth*imageHeight*imageChannels)*sizeof(unsigned char));
  cudaMalloc((void **)&dev_gray_uchar, (imageWidth*imageHeight*imageChannels)*sizeof(unsigned char));
  cudaMalloc((void **)&dev_gray_float, (imageWidth*imageHeight*imageChannels)*sizeof(float));
  
  cudaMemcpy(dev_float, hostInputImageData, (imageWidth*imageHeight*imageChannels)*sizeof(float), \
      cudaMemcpyHostToDevice);
  //@ 2. define kernel launch parameters
  dim3 grid_dim{(imageWidth - 1) / BLOCK_SIZE + 1, (imageHeight - 1) / BLOCK_SIZE + 1, 1};
  dim3 block_dim{BLOCK_SIZE, BLOCK_SIZE, 1};

  float2uchar<<<grid_dim, block_dim>>>(dev_uchar, dev_float, imageWidth, imageHeight, imageChannels);
  rgb2gray<<<grid_dim, block_dim>>>(dev_gray_uchar, dev_uchar, imageWidth, imageHeight,  imageChannels);
  
  cudaDeviceSynchronize();
  //@ 2. float to unsigned char
  
  //@ uchar to float
  uchar2float<<<grid_dim, block_dim>>>(dev_gray_float, dev_gray_uchar, imageWidth, imageHeight, imageChannels);
  //@ copy result from device back to host
  
  wbImage_t outputGrayImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  float *hostOutputGrayImageData = wbImage_getData(outputGrayImage);
  cudaMemcpy(hostOutputGrayImageData, dev_gray_float, (imageWidth*imageHeight*imageChannels)*sizeof(float), \
      cudaMemcpyDeviceToHost);
  
  wbSolution(args, outputGrayImage);

  // cudaMemcpy(hostOutputImageData, dev_float, (imageWidth*imageHeight*imageChannels)*sizeof(float), \
      cudaMemcpyDeviceToHost);
  
  cudaDeviceSynchronize();

  // wbSolution(args, outputImage);
  
  cudaFree(dev_float);
  cudaFree(dev_uchar);
  wbImage_delete(inputImage);
  wbImage_delete(outputImage);
  //@@ insert code here

  return 0;
}
