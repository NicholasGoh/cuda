#include <stdio.h>
#include <assert.h>

#define N 2048 * 2048 // Number of elements in each vector

__global__ void saxpy(int * a, int * b, int * c, int maxIndex)
{
  // Determine our unique global thread ID, so we know which element to process
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for (int i = tid; i < maxIndex; i += stride)
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
  int *a, *b, *c, *h_c;

  int size = N * sizeof (int); // The total number of bytes per vector

  int deviceId;
  int numberOfSMs;

  cudaCheck(cudaGetDevice(&deviceId));
  cudaCheck(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

  // Allocate memory
  cudaCheck(cudaMalloc(&a, size)); // gpu only
  cudaCheck(cudaMalloc(&b, size)); // gpu only
  cudaCheck(cudaMalloc(&c, size)); // gpu only
  cudaCheck(cudaMallocHost(&h_c, size)); // gpu only

  int threads_per_block = 256;
  int number_of_blocks = numberOfSMs * 32;
	const int numberOfSegments = 15;                  // This example demonstrates slicing the work into 4 segments.
	int segmentN = N / numberOfSegments;             // A value for a segment's worth of `N` is needed.

  // Initialize memory
	initWith <<<number_of_blocks, threads_per_block>>>(2, a);
	initWith <<<number_of_blocks, threads_per_block>>>(1, b);
	for (int i = 0; i < numberOfSegments; ++i){
		// Calculate the index where this particular segment should operate within the larger arrays.
	 	int offset = i * segmentN;
		
		// Create a stream for this segment's worth of copy and work.
		cudaStream_t stream;//, stream_cpy;
		cudaCheck(cudaStreamCreate(&stream));
//		cudaCheck(cudaStreamCreate(&stream_cpy));
		cudaEvent_t work;
		cudaCheck(cudaEventCreate(&work));

	  saxpy <<<number_of_blocks, threads_per_block, 0, stream>>>(&a[offset], &b[offset], &c[offset], segmentN);
		cudaCheck(cudaEventRecord(work, stream));
		// `cudaStreamDestroy` will return immediately (is non-blocking), but will not actually destroy stream until
		// all stream operations are complete.
		cudaCheck(cudaStreamDestroy(stream));

//		cudaCheck(cudaStreamWaitEvent(stream_cpy, work, 0));
//	  cudaCheck(cudaMemcpyAsync(&h_c[offset], &c[offset], size/segmentN, cudaMemcpyDeviceToHost, stream_cpy));
//		cudaCheck(cudaStreamDestroy(stream_cpy));
	}

  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize()); // Wait for the GPU to finish
	cudaCheck(cudaMemcpy(h_c, c, size, cudaMemcpyDeviceToHost));

  // Print out the first and last 5 values of c for a quality check
  for( int i = 0; i < 5; ++i )
    printf("h_c[%d] = %d, ", i, h_c[i]);
  printf ("\n");
  for( int i = N-5; i < N; ++i )
    printf("h_c[%d] = %d, ", i, h_c[i]);
  printf ("\n");
  // Free all our allocated memory
  cudaCheck(cudaFree(a)); cudaCheck(cudaFree(b)); cudaCheck(cudaFree(c)); cudaCheck(cudaFreeHost(h_c));
}
