/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

using namespace std;
///shared memory reduced min kernel
__global__ void shmem_min_kernel(float* d_out, const float* d_in, int size) {
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ float sdata_min[];
	if (myId<size) {
		sdata_min[tid] = d_in[myId];
	}
	else {
		sdata_min[tid] = 4.0;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
		if (tid < s) {
				sdata_min[tid] = fminf(sdata_min[tid],sdata_min[tid + s]);
		}
		__syncthreads();
	}
	__syncthreads();
	if (tid == 0) {
		d_out[blockIdx.x] = sdata_min[0];
		//printf("d_out[%d]=%d ", blockIdx.x, d_out[blockIdx.x]);
	}
	

}//shared memory reduced max kernel
__global__ void shmem_max_kernel(float* d_out, const float* d_in, int size) {
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ float sdata_max[];
	if (myId<size) {
		sdata_max[tid] = d_in[myId];
	}
	else {
		sdata_max[tid] = -4.0;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
		if (tid < s) {
			sdata_max[tid] = fmaxf(sdata_max[tid],sdata_max[tid + s]);
		}
		__syncthreads();
	}
	__syncthreads();
	if (tid == 0) {
		d_out[blockIdx.x] = sdata_max[0];
	}

}
//shared memory reduced exclusive scan kernel
__global__ void shmem_prefix_sum_kernel
(
	unsigned int* d_out,
	unsigned int* d_in,
	size_t n
)
{
	extern __shared__ unsigned int smdata[];
	int tid = threadIdx.x;
	int index = 2 * tid;
	int s;
	int l, r;
	int offset=1;
	

	//copy the data to the shared memory
	if (tid < n)
	{
		smdata[2 * tid] = d_in[2 * tid];
		smdata[2 * tid + 1] = d_in[2 * tid + 1];
	}
	__syncthreads();
	//up scan
	for (s = n / 2; s > 0; s /= 2) {
		__syncthreads();

		if (tid < s) {
			l = offset*(index + 1) - 1;
			r = offset*(index + 2) - 1;
			smdata[r] += smdata[l];
		}
		offset *= 2;
	}
	__syncthreads();
	if (tid == 0) {
		smdata[n - 1] = 0;
	}
	//down scan
	for (s = 1; s < n; s *= 2) {
		offset /= 2;
		__syncthreads();
		if (tid < s) {
			l = offset*(index + 1) - 1;
			r = offset*(index + 2) - 1;
			int temp = smdata[l];
			smdata[l] = smdata[r];
			smdata[r] += temp;
		}
	}
	__syncthreads();
	if (tid < n)
	{
		d_out[2 * tid] = smdata[2 * tid];
		d_out[2 * tid + 1] = smdata[2 * tid + 1];
	}
}
__global__
void shmem_generate_histogram
(
	const float* const d_logLuminance,
	unsigned int* histogram,
	float lumRange,
	float min_loglum,
	const size_t numBins,
	const size_t size
)
{
	int myId = blockIdx.x*blockDim.x + threadIdx.x;
	if (myId < size) {
		float lum = d_logLuminance[myId];
		int bin = int((lum - min_loglum) / lumRange * float(numBins));
		__syncthreads();
		atomicAdd(&histogram[bin], 1);
	}
}
__global__
void shmem_generate_hist_bin
(
	const float* const d_logLuminance,
	unsigned int* hist_bin,
	float lumRange,
	float min_loglum,
	const size_t numBins,
	const size_t size
)
{
	int myId = blockIdx.x*blockDim.x + threadIdx.x;
	if (myId < size) {
		float lum = d_logLuminance[myId];
		int bin = int((lum - min_loglum) / lumRange * float(numBins));
		__syncthreads();
		atomicAdd(&hist_bin[blockIdx.x*blockDim.x+bin], 1);
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
	__syncthreads();
	atomicAdd(&hist[tid], hist_bin[myId]);
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  //Here are the steps you need to implement
  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  int size = numRows * numCols;
  //cout << size << endl;
  //give enough blocks to process the data
  int remain;
  if (! size / maxThreadsPerBlock == 0) {
	  remain = 1;
  }
  int blocks = size / maxThreadsPerBlock + remain ;

  /*
  //reference to check whether the min kernel and max kernel is right
  float *h_logLuminance = new float[size];
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, size*sizeof(float), cudaMemcpyDeviceToHost));
  float max_v = h_logLuminance[0];
  float min_v = h_logLuminance[0];
  for (int i = 0; i < size; i++) {
	  min_v = fmin(min_v, h_logLuminance[i]);
	  max_v = fmax(max_v, h_logLuminance[i]);
	  if (-1.64725e+38 == h_logLuminance[i]) {
		  cout << "error?" << endl;
	  }
  }
  cout << max_v<<" "<<min_v<<endl;
  delete [] h_logLuminance;
  */

  //define data used
  float *d_min;
  float *d_max;
  float *d_min_among_block;
  float *d_max_among_block;
  unsigned int *d_hist;
  unsigned int *d_hist2;
  unsigned int *d_hist_bin;
  float range;

  //assign gpu memory to data
  checkCudaErrors(cudaMalloc((void**)&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_max, sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_hist, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**)&d_min_among_block, blocks * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_max_among_block, blocks * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_hist2, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**)&d_hist_bin, numBins * blocks * sizeof(unsigned int)));
  
  /*
  1) find the minimum and maximum value in the input logLuminance channel
  store in min_logLum and max_logLum
  */

  //compute the min values of each block
  shmem_min_kernel <<< blocks, threads, threads * sizeof(float) >>> (d_min_among_block, 
																	d_logLuminance,
																	size
																	);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  /*
  //check the min values of each block
  float *h_min_among_block = new float[blocks];
  checkCudaErrors(cudaMemcpy(h_min_among_block, d_min_among_block, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < blocks; i++) {
	  cout << h_min_among_block[i] << "/";
  }
  cout << endl;
  delete [] h_min_among_block;
  */

  //compute min values among all blocks
  shmem_min_kernel <<< 1, threads, threads * sizeof(float) >> > (d_min,
																  d_min_among_block,
																  blocks
																  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //compute the max values of each block
  shmem_max_kernel <<< blocks, threads, threads * sizeof(float) >>> (d_max_among_block,
																	d_logLuminance, 
																	size
																	);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
  //check the max values of each block
  float *h_max_among_block = new float[blocks];
  checkCudaErrors(cudaMemcpy(h_max_among_block, d_max_among_block, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < blocks; i++) {
	  cout << h_max_among_block[i] << "/";
  }
  cout << endl;
  delete [] h_max_among_block;
  */

  shmem_max_kernel <<< 1, threads, threads * sizeof(float) >> > (d_max,
																  d_max_among_block,
																  blocks
																  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  //copy the value into min_logLum and max_logLum
  checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

  /*
  //cout to see the difference
  cout<<min_logLum<<endl;
  cout << max_logLum << endl;
  */

  //2) subtract them to find the range

  range = max_logLum - min_logLum;
  //cout << range << endl;

  /*3) generate a histogram of all the values in the logLuminance channel using
  the formula : bin = (lum[i] - lumMin) / lumRange * numBins
  */

  //generate histogram of the image
  //compute kernel time for each method
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  //method 1 only use atom to add values to each bin
  shmem_generate_histogram <<< blocks, threads, threads * sizeof(float) >>> (d_logLuminance,
													d_hist,
													range,
													min_logLum,
													numBins,
													size
													);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Time for the kernel: %f ms\n", time);


  cudaEventRecord(start, 0);

  //method 2 first generate first atom computation the histogram of each block
  shmem_generate_hist_bin <<< blocks, threads, threads * sizeof(float) >>> (d_logLuminance,
	  d_hist_bin,
	  range,
	  min_logLum,
	  numBins,
	  size
	  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //then atom computation among those blocks
  shmem_combine_hist_bin <<< blocks, threads, threads * sizeof(unsigned int) >>> (d_hist_bin, d_hist2);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Time for the kernel2: %f ms\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  /*
  //check the value of histogram values of method 1 and method 2
  unsigned int *h_hist;
  h_hist= new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(h_hist, d_hist, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  unsigned int *h_hist2;
  h_hist2 = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(h_hist2, d_hist2, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < numBins; i++) {
		printf("%d/%d ", h_hist[i], h_hist2[i]);
  }
  printf("\n");
  */
  

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  /*
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  
  shmem_prefix_sum_kernel <<< blocks, threads / 2, 2*numBins * sizeof(unsigned int) >>> ( d_cdf, d_hist2, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
  //check the value of exclusive scan
  unsigned int *h_cdf;
  h_cdf = new unsigned int[numBins];
  cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numBins; i++) {
	  cout<<h_cdf[i]<<" ";
  }
  cout << endl;
  */

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //set free the memory
  
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));
  checkCudaErrors(cudaFree(d_min_among_block));
  checkCudaErrors(cudaFree(d_max_among_block));
  checkCudaErrors(cudaFree(d_hist));

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
