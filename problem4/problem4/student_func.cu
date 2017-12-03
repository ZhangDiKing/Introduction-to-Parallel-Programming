//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
using namespace std;
///shared memory reduced min kernel
//index if bit==1, index==1
__global__ void getNewIndex
(
	unsigned int *d_out, 
	unsigned int *d_in,
	unsigned int* d_cdf_one,
	unsigned int *d_cdf_zero,
	const unsigned int *d_one,
	unsigned int zeros_sum,
	int n
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int new_index=0;
	if (myId < n) {
		if (d_one[myId] == 1) {
			new_index = d_cdf_one[myId] + zeros_sum;
		}
		else {
			new_index = d_cdf_zero[myId];
		}
		__syncthreads();
		d_out[new_index] = d_in[myId];
	}
}
__global__ void getOnesZeros
(
	unsigned int* d_one, 
	unsigned int* d_zero,
	unsigned int* d_in, 
	int digit
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ unsigned int smdata[];
	int tid = threadIdx.x;
	smdata[tid] = (d_in[myId] >> digit);
	__syncthreads();
	if (smdata[tid] % 2==0) {
		d_zero[myId] = 1;
		d_one[myId] = 0;
	}
	else {
		d_one[myId] = 1;
		d_zero[myId] = 0;
	}
}
 //shared memory reduced exclusive scan kernel with different block
__global__ void shmem_prefix_sum_among_block
(
	unsigned int* d_out,
	unsigned int* d_in,
	size_t n
)
{
	extern __shared__ unsigned int smdata[];

	int myId = blockIdx.x * blockDim.x * 2;
	int tid = threadIdx.x;
	int index = 2 * tid;
	int s;
	int l, r;
	int offset = 1;


	//copy the data to the shared memory
	if(myId + 2 * tid  + 1<n){
		
		smdata[2 * tid + 1] = d_in[myId + 2 * tid + 1];
	}
	else {
		smdata[2 * tid + 1] = 0;
	}
	__syncthreads();
	if (myId + 2 * tid  <n) {
		smdata[2 * tid] = d_in[myId + 2 * tid];
	}
	else {
		smdata[2 * tid] = 0;
	}
	__syncthreads();
	//up scan
	for (s = blockDim.x; s > 0; s /= 2) {
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
		smdata[blockDim.x*2 - 1] = 0;
	}
	//down scan
	for (s = 1; s < blockDim.x*2; s *= 2) {
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
	if (myId + 2 * tid + 1<n)
	{
		d_out[myId + 2 * tid + 1] = smdata[2 * tid + 1];
	}
	__syncthreads();
	if (myId + 2 * tid <n) {
		d_out[myId + 2 * tid] = smdata[2 * tid];
	}
}
__global__ void shmem_sum_kernel(unsigned int* d_out, unsigned int* d_in, int size) {
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ unsigned int sdata_sum[];
	if (myId < size) {
		sdata_sum[tid] = d_in[myId];
	}
	else {
		sdata_sum[tid] = 0;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s =s/2) {
		if (tid < s) {
			sdata_sum[tid] +=sdata_sum[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();
	if (tid == 0) {
		d_out[blockIdx.x] = sdata_sum[tid];
		//printf("d_out[%d]=%d ", blockIdx.x, d_out[blockIdx.x]);
	}

}
__global__ void shmem_add_block_sum_scan
(
	unsigned int *d_in,
	unsigned int *sum,
	int n
) 
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId<n) {
		d_in[myId] = d_in[myId]+sum[blockIdx.x];
		__syncthreads();
	}
}

void your_sort(unsigned int* const d_inputVals,
	unsigned int* const d_inputPos,
	unsigned int* const d_outputVals,
	unsigned int* const d_outputPos,
	const size_t numElems)
{
	//TODO
	//PUT YOUR SORT HERE
	const int maxThreadsPerBlock = 1024;
	//const int maxThreadsPerBlock = 4;
	int threads = maxThreadsPerBlock;
	int size = numElems;
	//int size = 10;
	int blocks = size / threads;
	if (size%threads != 0) {
		blocks += 1;
	}
	cout << blocks << endl;
	cout << numElems << endl;
	//initialize variables
	//unsigned int *d_array;
	//unsigned int *d_pos;
	unsigned int *d_new_array;
	unsigned int *d_new_pos;
	unsigned int *d_out;
	unsigned int *d_cdf_one;
	unsigned int *d_cdf_zero;
	unsigned int *d_one;
	unsigned int *d_zero;
	unsigned int *d_sum;
	unsigned int *d_sum_block;
	unsigned int *d_one_sum;
	unsigned int *d_zero_sum;
	unsigned int zero_sum;
	int max_digit = 32;
	//checkCudaErrors(cudaMalloc((void**)&d_array, size * sizeof(unsigned int)));
	//checkCudaErrors(cudaMalloc((void**)&d_pos, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_new_array, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_new_pos, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_out, blocks * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_cdf_one, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_cdf_zero, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_one, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_one_sum, blocks * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_zero_sum, blocks * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_zero, size * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_sum, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_sum_block, blocks *sizeof(unsigned int)));
	//display the orignal array
	/*
	unsigned int h_array[10] = { 3,7,2,6,8,10,1,4,9,9 };
	unsigned int h_pos[10] = { 0,1,2,3,4,5,6,7,8,9 };
	
	unsigned int *h_temp_array=new unsigned int[size];
	unsigned int *h_temp_pos = new unsigned int[size];
	checkCudaErrors(cudaMemcpy(d_array, h_array, size * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pos, h_pos, size * sizeof(unsigned int), cudaMemcpyHostToDevice));
	*/
	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	for (int d = 0; d < max_digit; d++) {
		cout << d << endl;
		//getOnesZeros <<<blocks, threads, threads * sizeof(unsigned int) >> > (d_one, d_zero, d_array, d);
		getOnesZeros << <blocks, threads, threads * sizeof(unsigned int) >> > (d_one, d_zero, d_outputVals, d);
		//check the ones and zeros
		cout<<"zeros and ones"<<endl;
		unsigned int *h_one = new unsigned int[size];
		unsigned int *h_zero = new unsigned int[size];
		checkCudaErrors(cudaMemcpy(h_one, d_one, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_zero, d_zero, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		int count_zero=0;
		for (int i = 0; i < threads; i++) {
		//cout << h_zero[i] << "/";
		count_zero += h_zero[i];
		}
		//
		//cout << endl;
		
		//for (int i = 0; i < threads; i++) {
			//cout << h_one[i] << "/";
		//}
		//cout << endl;
		cout << count_zero << endl;
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//get cdf of zero
		shmem_prefix_sum_among_block << <blocks, threads/2, 2 * threads * sizeof(unsigned int) >> > (d_cdf_zero, d_zero, size);
		shmem_sum_kernel << <blocks, threads, threads * sizeof(unsigned int) >> > (d_sum_block, d_zero, size);
		shmem_prefix_sum_among_block << <1, threads/2, 2*threads * sizeof(unsigned int) >> > (d_zero_sum, d_sum_block, blocks);
		shmem_add_block_sum_scan << <blocks, threads, threads * sizeof(unsigned int) >> > (d_cdf_zero, d_zero_sum, size);

		//get sum of zero
		//shmem_sum_kernel << <blocks, threads, threads * sizeof(unsigned int) >> > (d_sum_block, d_zero, size);
		//check the sum among the blocks
		/*
		unsigned int *h_sum_block = new unsigned int[blocks];
		//checkCudaErrors(cudaMemcpy(h_sum_block, d_sum_block, blocks*sizeof(unsigned int), cudaMemcpyDeviceToHost));
		cudaMemcpy(h_sum_block, d_out, blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < blocks; i++) {
			cout << h_sum_block[i] << "/";
		}
		cout << endl;
		*/
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		shmem_sum_kernel << <2, threads, threads * sizeof(unsigned int) >> > (d_sum, d_sum_block, blocks);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		zero_sum = 0;
		//checkCudaErrors(cudaMemcpy(&zero_sum, d_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		cudaMemcpy(&zero_sum, d_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cout << zero_sum << endl;
		//get cdf of one
		shmem_prefix_sum_among_block << <blocks, threads / 2, 2 * threads * sizeof(unsigned int) >> > (d_cdf_one, d_one, size);
		shmem_sum_kernel << <blocks, threads, threads * sizeof(unsigned int) >> > (d_out, d_one, size);
		
		/*check the d_out 
		unsigned int *h_out = new unsigned int[blocks];
		checkCudaErrors(cudaMemcpy(h_out, d_out, blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < blocks; i++) {
		cout << h_out[i] << "/";
		}
		cout << endl;
		*/
		
		//shmem_prefix_sum_kernel
		shmem_prefix_sum_among_block << <1, threads/2, 2*threads * sizeof(unsigned int) >> > (d_one_sum, d_out, blocks);
		/*check the d_one_sum 
		unsigned int *h_one_sum = new unsigned int[blocks];
		checkCudaErrors(cudaMemcpy(h_one_sum, d_one_sum, blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < blocks; i++) {
			cout << h_one_sum[i] << "/";
		}
		cout << endl;
		*/
		shmem_add_block_sum_scan << <blocks, threads, threads * sizeof(unsigned int) >> > (d_cdf_one, d_one_sum, size);
		


		//check the cdf of zeros and ones
		/*
		unsigned int *h_cdf_one = new unsigned int[size];
		unsigned int *h_cdf_zero = new unsigned int[size];
		checkCudaErrors(cudaMemcpy(h_cdf_one, d_cdf_one, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_cdf_zero, d_cdf_zero, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		
		/*display the cdf of ones and zeros
		cout << "the cdf of ones and zeros are:" << endl;
		for (int i = 0; i < size; i++) {
			cout << h_cdf_one[i] << "/";
		}
		cout << endl;
		for (int i = 0; i < size; i++) {
			cout << h_cdf_zero[i] << "/";
		}
		cout << endl;
		getNewIndex << <blocks, threads, threads * sizeof(unsigned int) >> >
			(d_new_array, d_array, d_cdf_one, d_cdf_zero, d_one, zero_sum, size);

		getNewIndex << <blocks, threads, threads * sizeof(unsigned int) >> >
			(d_new_pos, d_pos, d_cdf_one, d_cdf_zero, d_one, zero_sum, size);

		checkCudaErrors(cudaMemcpy(d_array, d_new_array, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_pos, d_new_pos, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		*/
		getNewIndex << <blocks, threads, threads * sizeof(unsigned int) >> >
			(d_new_array, d_outputVals, d_cdf_one, d_cdf_zero, d_one, zero_sum, size);

		getNewIndex << <blocks, threads, threads * sizeof(unsigned int) >> >
			(d_new_pos, d_outputPos, d_cdf_one, d_cdf_zero, d_one, zero_sum, size);

		checkCudaErrors(cudaMemcpy(d_outputVals, d_new_array, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputPos, d_new_pos, size * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		/*
		cout << "updated array of "<<d<<"th digit is:"<<endl;
		checkCudaErrors(cudaMemcpy(h_temp_array, d_array, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < size; i++) {
			cout << h_temp_array[i] << "/";
		}
		cout << endl;
		*/
	}
	/*
	unsigned int *h_array=new unsigned int[size];
	cout << "updated array is:" << endl;
	checkCudaErrors(cudaMemcpy(h_array, d_outputVals, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < size; i++) {
	cout << h_array[i] << "/";
	}
	cout << endl;
	*/
	/*
	checkCudaErrors(cudaMemcpy(h_temp_pos, d_pos, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < size; i++) {
		cout << h_temp_pos[i] << "/";
	}
	cout << endl;
	*/
	//free the memory
	//checkCudaErrors(cudaFree(d_array));
	checkCudaErrors(cudaFree(d_new_array));
	checkCudaErrors(cudaFree(d_new_pos));
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_cdf_one));
	checkCudaErrors(cudaFree(d_cdf_zero));
	checkCudaErrors(cudaFree(d_one));
	checkCudaErrors(cudaFree(d_zero));
	checkCudaErrors(cudaFree(d_sum));
	checkCudaErrors(cudaFree(d_sum_block));
	checkCudaErrors(cudaFree(d_one_sum));
	checkCudaErrors(cudaFree(d_zero_sum));
	//your_prefixsum(d_array,d_cdf,numElems);
	//cout << size << endl;
	//give enough blocks to process the data

}