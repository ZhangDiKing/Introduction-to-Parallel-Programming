/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
__global__
void shmem_generate_hist_bin
(
	const unsigned int * d_in,
	unsigned int* hist_bin,
	const unsigned int numBins,
	const unsigned int numElems
)
{
	int myId = blockIdx.x*blockDim.x + threadIdx.x;
	extern __shared__ unsigned int smdata[];
	int tid = threadIdx.x;
	if (myId < numElems) {
		smdata[tid] = d_in[myId];
		__syncthreads();
		unsigned int bin = blockIdx.x*blockDim.x+smdata[tid];
		atomicAdd(&hist_bin[bin], 1);
	}
}
__global__
void shmem_combine_hist_bin
(
	unsigned int* hist_bin,
	unsigned int* hist
)
{
	int myId = blockIdx.x*blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ unsigned int smdata[];
	smdata[tid] = hist_bin[myId];
	__syncthreads();
	atomicAdd(&hist[tid], smdata[tid]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
	const int maxThreadsPerBlock = numBins;
	int threads = maxThreadsPerBlock;
	//cout << size << endl;
	//give enough blocks to process the data
	int remain;
	if (!numElems / maxThreadsPerBlock == 0) {
		remain = 1;
	}
	int blocks = numElems / maxThreadsPerBlock + remain;
	//printf("%d", blocks);

	unsigned int *d_hist_bin;
	checkCudaErrors(cudaMalloc((void**)&d_hist_bin, numBins * blocks * sizeof(unsigned int)));

	shmem_generate_hist_bin <<< blocks, threads, threads * sizeof(unsigned int) >> > (d_vals,d_hist_bin,numBins, numElems);
	shmem_combine_hist_bin <<< blocks, threads, threads * sizeof(unsigned int) >> > (d_hist_bin, d_histo);

  checkCudaErrors(cudaFree(d_hist_bin));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
