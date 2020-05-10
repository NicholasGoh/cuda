#include <stdio.h>
#include <assert.h>

#define N 2048 * 2048 // Number of elements in each vector

__global__ void saxpy(int * a, int * b, int * c)
{
  // Determine our unique global thread ID, so we know which element to process
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = tid; i < N; i += stride)
    c[i] = 2 * a[i] + b[i];
}

// init on gpu
__global__ void initWith(int value, int * a){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

	for (int i = tid; i< N; i += stride)
		a[i] = value;
}

// check error
inline cudaError_t cudaCheck(cudaError_t result){
	if (result!=cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}

int main()
{
  int *a, *b, *c;

  int size = N * sizeof (int); // The total number of bytes per vector

  int deviceId;
  int numberOfSMs;

  cudaCheck(cudaGetDevice(&deviceId));
  cudaCheck(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

  // Allocate memory
  cudaCheck(cudaMalloc(&a, size)); // gpu only
  cudaCheck(cudaMalloc(&b, size)); // gpu only
  cudaCheck(cudaMallocManaged(&c, size)); // both

  int threads_per_block = 256;
  int number_of_blocks = numberOfSMs * 32;

  // Initialize memory
	initWith <<<number_of_blocks, threads_per_block>>>(2, a);
	initWith <<<number_of_blocks, threads_per_block>>>(1, b);

  saxpy <<<number_of_blocks, threads_per_block>>>(a, b, c);

  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize()); // Wait for the GPU to finish

  // Print out the first and last 5 values of c for a quality check
  for( int i = 0; i < 5; ++i )
    printf("c[%d] = %d, ", i, c[i]);
  printf ("\n");
  for( int i = N-5; i < N; ++i )
    printf("c[%d] = %d, ", i, c[i]);
  printf ("\n");

  // Free all our allocated memory
  cudaCheck(cudaFree(a)); cudaCheck(cudaFree(b)); cudaCheck(cudaFree(c));
}
