
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>
#include <string>

//px = 480;py = 1;
//v210, byte per thread: 480/6 * 16 = 1280;
//dim3 gridsize(1, 68);
//dim3 blocksize(8, 32);
//input UHD V210
//ouput UHD V210
__global__ void Kernel_V210_Copy(
	char *ouput,
	char *input
)
{
	int subx = blockIdx.x * blockDim.x + threadIdx.x;//blockIdx.x * 8 + threadIdx.x;
	int suby = blockIdx.y * blockDim.y + threadIdx.y;//blockIdx.y * 32 + threadIdx.y

													 //int Imgx = subx * 480 + 0 ~480 - 1;//Imgx = Subx * Px + 0 ~ Px -1
													 //int Imgy = suby * 1 + 0 ~1 - 1 = suby;//Imgy = Suby * Py + 0 ~ Py -1

													 //v210 - > float

	if (suby >= 2160)
	{
		return;
	}

	_int8 *input_int8_pV210_10240_line = input + suby * 10240;
	_int8 *output_int8_pV210_10240_line = ouput + suby * 10240;

	//method1
	//for (int i = 0; i < 1280; i++)
	//{
	//	ouput[suby * 10240 + subx * 1280 + i] = input[suby * 10240 + subx * 1280 + i];
	//}

	//method2

	//for (int i = 0; i < 1280; i++)
	//{
	//	*(output_int8_pV210_10240_line + subx * 1280 + i) = *(input_int8_pV210_10240_line + subx * 1280 + i);
	//}

	//method3
	memcpy(output_int8_pV210_10240_line + subx * 1280, input_int8_pV210_10240_line + subx * 1280, 1280);

}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t V210_Copy(unsigned char  *ouput, unsigned char *input, long size)
{

	//LARGE_INTEGER frequency;
	//LARGE_INTEGER time0, time1, time2, time3, time4, time5;


	_int8 *dev_output = 0;
	_int8 *dev_input = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, size * sizeof(_int8), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	dim3 gridsize(1, 68);

	dim3 blocksize(8, 32);


	Kernel_V210_Copy << <gridsize, blocksize >> >(
		dev_output,
		dev_input
		);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(ouput, dev_output, size * sizeof(_int8), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);


	return cudaStatus;
}


int main()
{
   
	printf("Output YUV:\n");


	//// sample1 copy test
	FILE *file;
	errno_t er = fopen_s(&file, "QuadHD_V210.yuv", "rb");
	if (er != 0)
	{
		return 0;
	}
	long lFrameSize = 10240 * 2160;

	unsigned char *uchFile = new unsigned char[10240 * 2160];

	unsigned char *i8Input = new unsigned char[lFrameSize];
	unsigned char *i8Output = new unsigned char[lFrameSize];

	size_t rsize = fread_s(uchFile, 10240 * 2160, 1, 10240 * 2160, file);

	i8Input = (unsigned char *)uchFile;

	cudaDeviceProp dProp;
	cudaGetDeviceProperties(&dProp, 0);

	int count;

	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);



	V210_Copy(i8Output, i8Input, lFrameSize);

	FILE *wfile;
	er = fopen_s(&wfile, "QuadHD_V210_copy.yuv", "wb");
	fwrite((unsigned char *)i8Output, 1, 10240 * 2160, wfile);
	fflush(wfile);
	fclose(wfile);


    return 0;
}
