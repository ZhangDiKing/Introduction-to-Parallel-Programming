//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/sequence.h>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <vector>
using namespace std;
//generate the mask
__global__ void generate_mask
(
	unsigned char* d_mask, 
	const uchar4* const d_sourceImg, 
	int size
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__  unsigned char smdata[];
	int tid = threadIdx.x;
	if (myId < size) {
		smdata[tid * 3] = (unsigned char) d_sourceImg[myId].x;
		smdata[tid * 3 + 1] = (unsigned char) d_sourceImg[myId].y;
		smdata[tid * 3 + 2] = (unsigned char) d_sourceImg[myId].z;
	}
	__syncthreads();
	if (myId < size) {
		d_mask[myId] = 1 - (smdata[tid * 3] + smdata[tid * 3 + 1] + smdata[tid * 3 + 2]) / (255 * 3);
	}
}
//compute strictly interior pixels and border pixels
__global__ void compute_inter_border
(
	unsigned char * d_mask, 
	unsigned char * d_borderPixels, 
	unsigned char * d_strictInteriorPixels,
	int * d_strictInteriorPixels_int,
	int n_row,
	int n_col
) 
{
	int col = threadIdx.x;
	int row = blockIdx.x;
	extern __shared__  unsigned char smdata[];
	int id = threadIdx.x;
	if (row < n_row - 1 && row >= 1 && col <= n_col - 1 && col >= 1) {
		smdata[3 * id] = d_mask[(row - 1) * n_col + col - 1];
		smdata[3 * id + 1] = d_mask[row * n_col + col - 1];
		smdata[3 * id + 2] = d_mask[(row + 1) * n_col + col - 1];
		// boundary condition 
		if (id == n_col - 1) {
			smdata[3 * (id + 1)] = d_mask[(row - 1) * n_col + col];
			smdata[3 * (id + 1) + 1] = d_mask[row * n_col + col];
			smdata[3 * (id + 1) + 2] = d_mask[(row + 1) * n_col + col];
			smdata[3 * (id + 2)] = d_mask[(row - 1) * n_col + col + 1];
			smdata[3 * (id + 2) + 1] = d_mask[row * n_col + col + 1];
			smdata[3 * (id + 2) + 2] = d_mask[(row + 1) * n_col + col + 1];
		}
		__syncthreads();
		d_strictInteriorPixels[row * n_col + col] = smdata[3 * (id + 1) + 1] *
													smdata[3 * (id + 1)] *
													smdata[3 * (id + 1) + 2] *
													smdata[3 * (id + 2) + 1] *
													smdata[3 * id + 1];
		d_strictInteriorPixels_int[row * n_col + col] = (int)d_strictInteriorPixels[row * n_col + col];
		d_borderPixels[row * n_col + col] = smdata[3 * (id + 1) + 1] && (!(
											smdata[3 * (id + 1)] &&
											smdata[3 * (id + 1) + 2] &&
											smdata[3 * (id + 2) + 1] &&
											smdata[3 * id + 1]) );
	}	

}

__global__ void separate_channel
(
	uchar4* d_sourceImg,
	unsigned char* d_red_src,
	unsigned char* d_blue_src,
	unsigned char* d_green_src,
	int size
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId < size) {
		d_red_src[myId] = (unsigned char)d_sourceImg[myId].x;
		d_blue_src[myId] = (unsigned char)d_sourceImg[myId].y;
		d_green_src[myId] = (unsigned char)d_sourceImg[myId].z;
	}
}

//compute the gradient for each channel
__global__ void computeG
(
	float *d_g,
	unsigned char* channel,
	unsigned char * d_strictInteriorPixels,
	int n_row,
	int n_col
) 
{
	int col = threadIdx.x;
	int row = blockIdx.x;
	extern __shared__  unsigned char smdata[];
	int id = threadIdx.x;
	if (row < n_row - 1 && row >= 1 && col <= n_col - 1 && col >= 1) {
		smdata[3 * id] = channel[(row - 1) * n_col + col - 1];
		smdata[3 * id + 1] = channel[row * n_col + col - 1];
		smdata[3 * id + 2] = channel[(row + 1) * n_col + col - 1];
		// boundary condition 
		if (id == n_col - 1) {
			smdata[3 * (id + 1)] = channel[(row - 1) * n_col + col];
			smdata[3 * (id + 1) + 1] = channel[row * n_col + col];
			smdata[3 * (id + 1) + 2] = channel[(row + 1) * n_col + col];
			smdata[3 * (id + 2)] = channel[(row - 1) * n_col + col + 1];
			smdata[3 * (id + 2) + 1] = channel[row * n_col + col + 1];
			smdata[3 * (id + 2) + 2] = channel[(row + 1) * n_col + col + 1];
		}
		__syncthreads();
		if (d_strictInteriorPixels[row * n_col + col]*1) {
			
			d_g[row * n_col + col] = 4.f * smdata[3 * (id + 1) + 1]-
									( (float) smdata[3 * (id + 1)] +
									(float) smdata[3 * (id + 1) + 2] +
									(float) smdata[3 * (id + 2) + 1] +
									(float) smdata[3 * id + 1]);
			
		}
	}

}
//transfer data type in preprocessing
__global__ void copy_uchar_to_float
(
	unsigned char *d_in_red,
	unsigned char *d_in_blue,
	unsigned char *d_in_green,
	float *d_out_red,
	float *d_out_blue,
	float *d_out_green,
	int n_col,
	int size
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId < size) {
		d_out_red[myId] = d_in_red[myId];
		d_out_blue[myId] = d_in_blue[myId];
		d_out_green[myId] = d_in_green[myId];
	}
}

__global__ void computeIteration
(
	const unsigned char* const d_dstImg,
	const unsigned char* const d_strictInteriorPixels,
	const unsigned char* const d_borderPixels,
	const int * d_index,
	const float* const d_f,
	const float* const d_g,
	float* const d_f_next,
	int size,
	int n_col
)
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId < size) {
		int offset = d_index[myId];
		float blendedSum = 0.f;
		float borderSum = 0.f;
		blendedSum = d_f[offset - 1] * d_strictInteriorPixels[offset - 1] +
					 d_f[offset + 1] * d_strictInteriorPixels[offset + 1] +
					 d_f[offset - n_col] * d_strictInteriorPixels[offset - n_col] +
					 d_f[offset + n_col] * d_strictInteriorPixels[offset + n_col];
		borderSum = d_dstImg[offset - 1] * (1 - d_strictInteriorPixels[offset - 1]) +
					d_dstImg[offset + 1] * (1 - d_strictInteriorPixels[offset + 1]) +
					d_dstImg[offset - n_col] * (1 - d_strictInteriorPixels[offset - n_col]) +
					d_dstImg[offset + n_col] * (1 - d_strictInteriorPixels[offset + n_col]);
		float f_next_val = (blendedSum + borderSum + d_g[offset]) / 4.f;
		d_f_next[offset] = min(255.f, max(0.f, f_next_val));

	}


}
__global__ void final_blend
(
	uchar4* d_blendedImg,
	float* d_blendedValsRed_2,
	float* d_blendedValsBlue_2,
	float* d_blendedValsGreen_2,
	const int * d_index,
	int size,
	int n_col
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId < size) {
		int offset = d_index[myId];
		d_blendedImg[offset].x = d_blendedValsRed_2[offset];
		d_blendedImg[offset].y = d_blendedValsBlue_2[offset];
		d_blendedImg[offset].z = d_blendedValsGreen_2[offset];
	}
}
void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{


	const int maxThreadsPerBlock = 1024;
	//const int maxThreadsPerBlock = 4;
	int threads = maxThreadsPerBlock;
	int size = numRowsSource * numColsSource;
	//int size = 10;
	int blocks = size / threads;
	/*configure 2d thread
	int th_x = 32;
	int th_y = 32;
	int block_y = numRowsSource / th_y;
	int block_x = numColsSource / th_x;
	if (numRowsSource % threads != 0) {
	block_x += 1;
	}
	if (numColsSource % threads != 0) {
	block_y += 1;
	}
	cout << block_x << endl;
	cout << block_y << endl;
	*/
	if (size%threads != 0) {
		blocks += 1;
	}
	//cout << numRowsSource << endl;
	//cout << numColsSource << endl;

	int thread_for_col = pow(2.0, (int) log2(numColsSource) + 1);
	//cout << thread_for_col << endl;
	//cout << blocks << endl;
	//cout << size << endl;

	unsigned char* d_mask;
	uchar4* d_sourceImg;
	uchar4* d_destImg;
	unsigned char *d_borderPixels;
	unsigned char *d_strictInteriorPixels;
	int *d_strictInteriorPixels_int;

	unsigned char* d_red_src;
	unsigned char* d_blue_src;
	unsigned char* d_green_src;

	unsigned char* d_red_dst;
	unsigned char* d_blue_dst;
	unsigned char* d_green_dst;

	float *d_g_red;
	float *d_g_blue;
	float *d_g_green;

	float *d_blendedValsRed_1;
	float *d_blendedValsRed_2;

	float *d_blendedValsBlue_1;
	float *d_blendedValsBlue_2;

	float *d_blendedValsGreen_1;
	float *d_blendedValsGreen_2;

	//memory allocation of device data
	cudaMalloc((void**)&d_mask, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_borderPixels, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_strictInteriorPixels, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_strictInteriorPixels_int, size * sizeof(int));

	cudaMalloc((void**)&d_sourceImg, size * sizeof(uchar4));
	cudaMalloc((void**)&d_destImg, size * sizeof(uchar4));

	cudaMalloc((void**)&d_red_src, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_green_src, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_blue_src, size * sizeof(unsigned char));

	cudaMalloc((void**)&d_red_dst, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_green_dst, size * sizeof(unsigned char));
	cudaMalloc((void**)&d_blue_dst, size * sizeof(unsigned char));

	cudaMalloc((void**)&d_g_red, size * sizeof(float));
	cudaMalloc((void**)&d_g_blue, size * sizeof(float));
	cudaMalloc((void**)&d_g_green, size * sizeof(float));

	cudaMalloc((void**)&d_blendedValsRed_1, size * sizeof(float));
	cudaMalloc((void**)&d_blendedValsBlue_1, size * sizeof(float));
	cudaMalloc((void**)&d_blendedValsGreen_1, size * sizeof(float));

	cudaMalloc((void**)&d_blendedValsRed_2, size * sizeof(float));
	cudaMalloc((void**)&d_blendedValsBlue_2, size * sizeof(float));
	cudaMalloc((void**)&d_blendedValsGreen_2, size * sizeof(float));

	//memory copy host to device
	cudaMemcpy(d_sourceImg, h_sourceImg, size * sizeof(uchar4), cudaMemcpyHostToDevice);
	cudaMemcpy(d_destImg, h_destImg, size * sizeof(uchar4), cudaMemcpyHostToDevice);

	/* To Recap here are the steps you need to implement

	1) Compute a mask of the pixels from the source image to be copied
	The pixels that shouldn't be copied are completely white, they
	have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
	*/
	generate_mask <<<blocks, threads, threads * 3 * sizeof(unsigned char) >>> 
		(d_mask, d_sourceImg, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	/*
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
	*/
	
	compute_inter_border <<<numRowsSource, thread_for_col, (thread_for_col +2) * 3 * sizeof(unsigned char) >> >
		(
			d_mask,
			d_borderPixels,
			d_strictInteriorPixels,
			d_strictInteriorPixels_int,
			numRowsSource,
			numColsSource
		);
	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());
	
	//use the function of thrust to get non-zero positions
	thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_strictInteriorPixels_int);
	thrust::device_vector<int> stencil(dev_ptr, dev_ptr+size);
	// storage for the nonzero indices
	thrust::device_vector<int> indices(size);

	// counting iterators define a sequence [0, 8)
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + size;

	// compute indices of nonzero elements 
	typedef thrust::device_vector<int>::iterator IndexIterator;

	IndexIterator indices_end = thrust::copy_if(first, last,
		stencil.begin(),
		indices.begin(),
		thrust::identity<int>());
	int * d_index= thrust::raw_pointer_cast(indices.data());
	/*
		 3) Separate out the incoming image into three separate channels
	*/
	separate_channel<<<blocks,threads>>>
	(
		d_sourceImg,
		d_red_src,
		d_blue_src,
		d_green_src,
		size
		);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	separate_channel << <blocks, threads >> >
		(
			d_destImg,
			d_red_dst,
			d_blue_dst,
			d_green_dst,
			size
			);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	/*
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
	*/
	computeG << < numRowsSource, thread_for_col, (thread_for_col + 2) * 3 * sizeof(unsigned char) >> >
		(
			d_g_red,
			d_red_src,
			d_strictInteriorPixels,
			numRowsSource,
			numColsSource
			);
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
	computeG << < numRowsSource, thread_for_col, (thread_for_col + 2) * 3 * sizeof(unsigned char) >> >
		(
			d_g_blue,
			d_blue_src,
			d_strictInteriorPixels,
			numRowsSource,
			numColsSource
			);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	computeG << < numRowsSource, thread_for_col, (thread_for_col + 2) * 3 * sizeof(unsigned char) >> >
		(
			d_g_green,
			d_green_src,
			d_strictInteriorPixels,
			numRowsSource,
			numColsSource
			);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//memory copy to the 2 buffers
	copy_uchar_to_float << <blocks, threads >> >
		(
			d_red_src,
			d_blue_src,
			d_green_src,
			d_blendedValsRed_1,
			d_blendedValsBlue_1,
			d_blendedValsGreen_1,
			numColsSource,
			size
			);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	copy_uchar_to_float << <blocks, threads >> >
		(
			d_red_src,
			d_blue_src,
			d_green_src,
			d_blendedValsRed_2,
			d_blendedValsBlue_2,
			d_blendedValsGreen_2,
			numColsSource,
			size
			);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	/*
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
	*/
	const size_t numIterations = 800;
	int index_size = (indices_end - indices.begin());
	int index_block = index_size / threads + 1;

	//for the red channel
	for (size_t i = 0; i < numIterations; ++i) {
		computeIteration << <index_block, threads >> >
			(
				d_red_dst,
				d_strictInteriorPixels,
				d_borderPixels,
				d_index,
				d_blendedValsRed_1,
				d_g_red,
				d_blendedValsRed_2,
				index_size,
				numColsSource
				);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		std::swap(d_blendedValsRed_1, d_blendedValsRed_2);
	}

	//for the blue channel
	for (size_t i = 0; i < numIterations; ++i) {
		computeIteration << <index_block, threads >> >
			(
				d_blue_dst,
				d_strictInteriorPixels,
				d_borderPixels,
				d_index,
				d_blendedValsBlue_1,
				d_g_blue,
				d_blendedValsBlue_2,
				index_size,
				numColsSource
				);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		std::swap(d_blendedValsBlue_1, d_blendedValsBlue_2);
	}
	//for the green channel
	for (size_t i = 0; i < numIterations; ++i) {
		computeIteration << <index_block, threads >> >
			(
				d_green_dst,
				d_strictInteriorPixels,
				d_borderPixels,
				d_index,
				d_blendedValsGreen_1,
				d_g_green,
				d_blendedValsGreen_2,
				index_size,
				numColsSource
				);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2);
	}

	std::swap(d_blendedValsRed_1, d_blendedValsRed_2);   //put output into _2
	std::swap(d_blendedValsBlue_1, d_blendedValsBlue_2);  //put output into _2
	std::swap(d_blendedValsGreen_1, d_blendedValsGreen_2); //put output into _2
	
	/*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
	*/
	final_blend<<<index_block, threads>>>
		(
		d_destImg,
		d_blendedValsRed_2,
		d_blendedValsBlue_2,
		d_blendedValsGreen_2,
		d_index,
		index_size,
		numColsSource
		);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	cudaMemcpy(h_blendedImg, d_destImg, size * sizeof(float), cudaMemcpyDeviceToHost);

      /*Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
	//free device data
	cudaFree(d_mask);
	cudaFree(d_sourceImg);
	cudaFree(d_blendedValsRed_1);
	cudaFree(d_blendedValsRed_2);
	cudaFree(d_blendedValsBlue_1);
	cudaFree(d_blendedValsBlue_2);
	cudaFree(d_blendedValsGreen_1);
	cudaFree(d_blendedValsGreen_2);
	cudaFree(d_g_red);
	cudaFree(d_g_blue);
	cudaFree(d_g_green);
	cudaFree(d_red_src);
	cudaFree(d_red_dst);
	cudaFree(d_blue_src);
	cudaFree(d_blue_dst);
	cudaFree(d_green_src);
	cudaFree(d_green_dst);
	cudaFree(d_borderPixels);
	cudaFree(d_strictInteriorPixels);
	//cudaFree(d_strictInteriorPixels_int);
	//cudaFree(d_index);
	//wow, we allocated a lot of memory!
}
