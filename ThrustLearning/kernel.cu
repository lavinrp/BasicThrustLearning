
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\device_ptr.h>
#include <thrust\memory.h>
#include <thrust\copy.h>

#include <stdio.h>

#include <iostream>



struct isEven {

	__device__ __host__
	bool operator()(int x) 
	{
		return (x % 2) == 0;
	}

};

__device__
int getGlobalIdx_3D_3D() 
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__global__
void addOne( int const * const __restrict__ inputData, int* __restrict__ outputData, size_t dataSize) 
{
	int thid = getGlobalIdx_3D_3D();
	if (thid >= dataSize) 
	{
		return;
	}

	outputData[thid] = inputData[thid] + 1;
}

void addOneExample(const int DATA_SIZE) 
{
	thrust::host_vector<int> h_inputData(DATA_SIZE);
	std::cout << "input data:" << std::endl;
	for (size_t i = 0; i < DATA_SIZE; i++) {
		h_inputData[i] = i;
		std::cout << i << std::endl;
	}
	std::cout << std::endl << std::endl;

	//move data to device
	thrust::device_vector<int> d_inputData = h_inputData;

	//allocate output data array
	thrust::device_vector<int> d_outputData(DATA_SIZE);

	addOne <<<2, 5>>>(thrust::raw_pointer_cast(d_inputData.data()), thrust::raw_pointer_cast(d_outputData.data()), DATA_SIZE);

	//move data back to host
	thrust::host_vector<int> h_outputData = d_outputData;

	std::cout << "output data: " << std::endl;
	for (size_t i = 0; i < h_outputData.size(); i++) {
		std::cout << h_outputData[i] << std::endl;
	}

	std::cout << std::endl << std::endl;
}

void filterExample(const int DATA_SIZE) 
{
	thrust::host_vector<int> h_inputData(DATA_SIZE);
	std::cout << "input data:" << std::endl;
	for (size_t i = 0; i < DATA_SIZE; i++) {
		h_inputData[i] = i;
		std::cout << i << std::endl;
	}
	std::cout << std::endl << std::endl;

	//move data to device
	thrust::device_vector<int> d_inputData = h_inputData;

	//allocate output data array
	thrust::device_vector<int> d_outputData(DATA_SIZE);

	auto lastCopiedValue = thrust::copy_if(d_inputData.begin(), d_inputData.end(), d_outputData.begin(), isEven());

	//move data back to host
	thrust::host_vector<int> h_outputData = d_outputData;

	std::cout << "output data: " << std::endl;
	for (int i = 0; i < std::distance(d_outputData.begin(), lastCopiedValue); i++) 
	{
		std::cout << h_outputData[i] << std::endl;
	}

	std::cout << std::endl << std::endl;

}

int main()
{

	//add one to data

	//create input data
	const int DATA_SIZE = 5;
	
	std::cout << "starting addOne" << std::endl;
	addOneExample(DATA_SIZE);

	std::cout << "press ENTER to continue...";
	std::cin.get();

	std::cout << "starting filter" << std::endl;
	filterExample(DATA_SIZE);

	std::cout << "press ENTER to exit...";
	std::cin.get();

	//filter data based on even odd


}

