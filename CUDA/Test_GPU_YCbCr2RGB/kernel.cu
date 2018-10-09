
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

#include "omp.h"

#include<chrono>

using namespace std;

using namespace std::chrono;



// calculate 10bit y from 32bit r, g, b
__device__ int GET_10Bit_R_FROM_10Bit_YUV_2020(unsigned int y, unsigned int u, unsigned int v)
{
	float r;
	r = 1.16438 * (y - 64.0f) + 1.67867 * (v - 512.0f) + 0.5f;

	return r;
}
// calculate 10bit u from 32bit r, g, b
__device__ int GET_10Bit_G_FROM_10Bit_YUV_2020(unsigned int y, unsigned int u, unsigned int v)
{
	float g;
	g = 1.16438 * (y - 64.0) - 0.18733 * (u - 512.0) - 0.65042 * (v - 512.0) + 0.5f;

	return g;
}
// calculate 10bit v from 32bit r, g, b
__device__ int GET_10Bit_B_FROM_10Bit_YUV_2020(unsigned int y, unsigned int u, unsigned int v)
{
	float b;
	b = 1.16438 * (y - 64.0) + 2.14177 * (u - 512.0) + 0.5f;

	return b;
}

//--------------

__device__ int GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange(unsigned int y, unsigned int u, unsigned int v)
{
	float r;
	r = (y)+1.4746 * (v - 512.0f) + 0.5f;

	return r;
}
// calculate 10bit u from 32bit r, g, b
__device__ int GET_10Bit_G_FROM_10Bit_YUV_2020_FullRange(unsigned int y, unsigned int u, unsigned int v)
{
	float g = 0.0;
	g = (y)-0.16455 * (u - 512.0) - 0.57135 * (v - 512.0) + 0.5f;

	return g;
}

// calculate 10bit v from 32bit r, g, b
__device__ int GET_10Bit_B_FROM_10Bit_YUV_2020_FullRange(unsigned int y, unsigned int u, unsigned int v)
{
	float b;
	b = (y)+1.8814 * (u - 512.0) + 0.5f;

	return b;
}



 int GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(unsigned int y, unsigned int u, unsigned int v)
{
	float r;
	r = (y)+1.4746 * (v - 512.0f) + 0.5f;

	return r;
}
// calculate 10bit u from 32bit r, g, b
int GET_10Bit_G_FROM_10Bit_YUV_2020_FullRange_CPU(unsigned int y, unsigned int u, unsigned int v)
{
	float g = 0.0;
	g = (y)-0.16455 * (u - 512.0) - 0.57135 * (v - 512.0) + 0.5f;

	return g;
}

// calculate 10bit v from 32bit r, g, b
int GET_10Bit_B_FROM_10Bit_YUV_2020_FullRange_CPU(unsigned int y, unsigned int u, unsigned int v)
{
	float b;
	b = (y)+1.8814 * (u - 512.0) + 0.5f;

	return b;
}



//-----------


// calculate 10bit y from 32bit r, g, b
__device__ int GET_10Bit_R_FROM_10Bit_YUV_709(unsigned int y, unsigned int u, unsigned int v)
{
	float r;
	r = 1.16438 * (y - 64.0f) + 1.79274 * (v - 512.0f) + 0.5f;

	return r;
}
// calculate 10bit u from 32bit r, g, b
__device__ int GET_10Bit_G_FROM_10Bit_YUV_709(unsigned int y, unsigned int u, unsigned int v)
{
	float g;
	g = 1.16438 * (y - 64.0) - 0.21325 * (u - 512.0) - 0.53291 * (v - 512.0) + 0.5f;

	return g;
}
// calculate 10bit v from 32bit r, g, b
__device__ int GET_10Bit_B_FROM_10Bit_YUV_709(unsigned int y, unsigned int u, unsigned int v)
{
	float b;
	b = 1.16438 * (y - 64.0) + 2.11240 * (u - 512.0) + 0.5f;

	return b;
}





//-------------
//px = 480;py = 1;
//byte per 480pixel of v210  =480/6 * 16 = 1280
//so, v210, byte per thread = 1280;
//dim3 gridsize(1, 68);
//dim3 blocksize(8, 32);
//input UHD V210
//ouput UHD V210
__global__ void Kernel_V210_2_RGB888(
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

	_int64 *pV210 = (_int64 *)((input + suby * 10240) + subx * 1280);
	_int8 *output_int8_RGB_11520_line = ouput + suby * 3840 * 3 + subx * 1440;//rgb888

	_int8 *output_int8_RGB = output_int8_RGB_11520_line;

	_int32 y[6];
	_int32 u[3];
	_int32 v[3];
	_int8 r[6];
	_int8 g[6];
	_int8 b[6];

	// a group  contain 6pixe,and contain 16 byte.
	for (int i = 0; i < 480 / 6; i++)
	{
		//V210 Format note:
		/*
		*pU = (*pV210) & 0x3FF;
		*pY = ( (*pV210)>>10 ) & 0x3FF;
		*pV = ( (*pV210)>>20 ) & 0x3FF;

		*(pY + 1)  = ( (*pV210)>>32 ) & 0x3FF;
		*(pU + 1)  = ( (*pV210)>>42 ) & 0x3FF;
		*(pY + 2)  = ( (*pV210)>>52 ) & 0x3FF;


		pV210 = pV210 + 1;

		*(pV + 1)  = ( (*pV210)     ) & 0x3FF;
		*(pY + 3)  = ( (*pV210)>>10 ) & 0x3FF;
		*(pU + 2)  = ( (*pV210)>>20 ) & 0x3FF;


		*(pY + 4)  = ( (*pV210)>>32 ) & 0x3FF;
		*(pV + 2)  = ( (*pV210)>>42 ) & 0x3FF;
		*(pY + 5)  = ( (*pV210)>>52 ) & 0x3FF;
		*/


		//*pU = (*pV210) & 0x3FF;
		u[0] = (*pV210) & 0x3FF;
		//*pY = ( (*pV210)>>10 ) & 0x3FF;
		y[0] = ((*pV210) >> 10) & 0x3FF;
		//*pV = ((*pV210) >> 20) & 0x3FF;
		v[0] = ((*pV210) >> 20) & 0x3FF;

		//*(pY + 1)  = ( (*pV210)>>32 ) & 0x3FF;
		y[1] = ((*pV210) >> 32) & 0x3FF;
		//*(pU + 1)  = ( (*pV210)>>42 ) & 0x3FF;
		u[1] = ((*pV210) >> 42) & 0x3FF;
		//*(pY + 2)  = ( (*pV210)>>52 ) & 0x3FF;
		y[2] = ((*pV210) >> 52) & 0x3FF;


		pV210 = pV210 + 1;

		//*(pV + 1)  = ( (*pV210)     ) & 0x3FF;
		v[1] = ((*pV210)) & 0x3FF;
		//*(pY + 3)  = ( (*pV210)>>10 ) & 0x3FF;
		y[3] = ((*pV210) >> 10) & 0x3FF;
		//*(pU + 2)  = ( (*pV210)>>20 ) & 0x3FF;
		u[2] = ((*pV210) >> 20) & 0x3FF;


		//*(pY + 4)  = ( (*pV210)>>32 ) & 0x3FF;
		y[4] = ((*pV210) >> 32) & 0x3FF;
		//*(pV + 2)  = ( (*pV210)>>42 ) & 0x3FF;
		v[2] = ((*pV210) >> 42) & 0x3FF;
		//*(pY + 5)  = ( (*pV210)>>52 ) & 0x3FF;
		y[5] = ((*pV210) >> 52) & 0x3FF;

		pV210 = pV210 + 1;

		//for (int i = 0; i < 6; i++)
		//{
		//	if (y[i] < 64)y[i] = 64;

		//	if (y[i] > 940)y[i] = 940;
		//}


		//for (int i = 0; i < 3; i++)
		//{
		//	if (u[i] < 64)u[i] = 64;
		//	if (v[i] < 64)v[i] = 64;

		//	if (u[i] > 960)u[i] = 960;
		//	if (v[i] > 960)v[i] = 960;
		//}



		_int8 iIndex = 0;

		for (int i = 0; i < 3; i++)
		{
			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;
			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;


			r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange(y[iIndex], u[i], v[i]) >> 2);
			g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_2020_FullRange(y[iIndex], u[i], v[i]) >> 2);
			b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_2020_FullRange(y[iIndex], u[i], v[i]) >> 2);

			iIndex++;
			r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange(y[iIndex], u[i], v[i]) >> 2);
			g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_2020_FullRange(y[iIndex], u[i], v[i]) >> 2);
			b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_2020_FullRange(y[iIndex], u[i], v[i]) >> 2);

			iIndex++;


			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;
			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;

		}

		(*output_int8_RGB) = r[0];
		*(output_int8_RGB + 1) = g[0];
		*(output_int8_RGB + 2) = b[0];
		*(output_int8_RGB + 3) = r[1];
		*(output_int8_RGB + 4) = g[1];
		*(output_int8_RGB + 5) = b[1];
		*(output_int8_RGB + 6) = r[2];
		*(output_int8_RGB + 7) = g[2];
		*(output_int8_RGB + 8) = b[2];
		*(output_int8_RGB + 9) = r[3];
		*(output_int8_RGB + 10) = g[3];
		*(output_int8_RGB + 11) = b[3];
		*(output_int8_RGB + 12) = r[4];
		*(output_int8_RGB + 13) = g[4];
		*(output_int8_RGB + 14) = b[4];
		*(output_int8_RGB + 15) = r[5];
		*(output_int8_RGB + 16) = g[5];
		*(output_int8_RGB + 17) = b[5];

		output_int8_RGB += 18;
	}
}


class Timer
{

public:

	Timer() : m_begin(high_resolution_clock::now()) {}

	void reset() { m_begin = high_resolution_clock::now(); }

	//默认输出毫秒
	int64_t elapsed() const

	{
		return duration_cast<chrono::milliseconds>(high_resolution_clock::now() - m_begin).count();
	}



	//微秒

	int64_t elapsed_micro() const

	{
		return duration_cast<chrono::microseconds>(high_resolution_clock::now() - m_begin).count();
	}



	//纳秒

	int64_t elapsed_nano() const

	{
		return duration_cast<chrono::nanoseconds>(high_resolution_clock::now() - m_begin).count();
	}



	//秒

	int64_t elapsed_seconds() const

	{
		return duration_cast<chrono::seconds>(high_resolution_clock::now() - m_begin).count();
	}



	//分

	int64_t elapsed_minutes() const

	{
		return duration_cast<chrono::minutes>(high_resolution_clock::now() - m_begin).count();
	}



	//时

	int64_t elapsed_hours() const

	{
		return duration_cast<chrono::hours>(high_resolution_clock::now() - m_begin).count();
	}
private:

	time_point<high_resolution_clock> m_begin;
};


// Helper function for using CUDA to add vectors in parallel.
cudaError_t V210_2_RGB888_GPU(unsigned char *input, long inputsize, unsigned char  *ouput, long outsize)
{



	Timer t;
	_int8 *dev_output = 0;
	_int8 *dev_input = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	t.reset();
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input, inputsize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_output, outsize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	printf("malloc GPU Memory %d ms\n", t.elapsed());

	t.reset();
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input, input, inputsize * sizeof(_int8), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	printf("upload input data:%d ms\n", t.elapsed());
	dim3 gridsize(1, 68);

	dim3 blocksize(8, 32);



	t.reset();

	double starttime, stoptime;
	starttime = omp_get_wtime();
	Kernel_V210_2_RGB888 << <gridsize, blocksize >> >(
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
	stoptime = omp_get_wtime();
	printf("convert: %d ms\n", t.elapsed());
	printf("convert: %3.2f ms\n", (stoptime - starttime)*1000);
	t.reset();
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(ouput, dev_output, outsize * sizeof(_int8), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	printf("dowload output data:%d ms\n", t.elapsed());
Error:
	cudaFree(dev_input);
	cudaFree(dev_output);

	return cudaStatus;
}




// Helper function for using CUDA to add vectors in parallel.
int V210_2_RGB888_CPU(unsigned char *input, long inputsize, unsigned char  *ouput, long outsize)
{



	_int64 *pV210 = (_int64 *)input;
	_int8 *output_int8_RGB_11520_line = (_int8 *)ouput;

	_int8 *output_int8_RGB = output_int8_RGB_11520_line;

	_int32 y[6];
	_int32 u[3];
	_int32 v[3];
	_int8 r[6];
	_int8 g[6];
	_int8 b[6];

	for (int h = 0; h < 2160; h++)
	{

	
	// a group  contain 6pixe,and contain 16 byte.
	for (int i = 0; i < 3840 / 6; i++)
	{
		//V210 Format note:
		/*
		*pU = (*pV210) & 0x3FF;
		*pY = ( (*pV210)>>10 ) & 0x3FF;
		*pV = ( (*pV210)>>20 ) & 0x3FF;

		*(pY + 1)  = ( (*pV210)>>32 ) & 0x3FF;
		*(pU + 1)  = ( (*pV210)>>42 ) & 0x3FF;
		*(pY + 2)  = ( (*pV210)>>52 ) & 0x3FF;


		pV210 = pV210 + 1;

		*(pV + 1)  = ( (*pV210)     ) & 0x3FF;
		*(pY + 3)  = ( (*pV210)>>10 ) & 0x3FF;
		*(pU + 2)  = ( (*pV210)>>20 ) & 0x3FF;


		*(pY + 4)  = ( (*pV210)>>32 ) & 0x3FF;
		*(pV + 2)  = ( (*pV210)>>42 ) & 0x3FF;
		*(pY + 5)  = ( (*pV210)>>52 ) & 0x3FF;
		*/


		//*pU = (*pV210) & 0x3FF;
		u[0] = (*pV210) & 0x3FF;
		//*pY = ( (*pV210)>>10 ) & 0x3FF;
		y[0] = ((*pV210) >> 10) & 0x3FF;
		//*pV = ((*pV210) >> 20) & 0x3FF;
		v[0] = ((*pV210) >> 20) & 0x3FF;

		//*(pY + 1)  = ( (*pV210)>>32 ) & 0x3FF;
		y[1] = ((*pV210) >> 32) & 0x3FF;
		//*(pU + 1)  = ( (*pV210)>>42 ) & 0x3FF;
		u[1] = ((*pV210) >> 42) & 0x3FF;
		//*(pY + 2)  = ( (*pV210)>>52 ) & 0x3FF;
		y[2] = ((*pV210) >> 52) & 0x3FF;


		pV210 = pV210 + 1;

		//*(pV + 1)  = ( (*pV210)     ) & 0x3FF;
		v[1] = ((*pV210)) & 0x3FF;
		//*(pY + 3)  = ( (*pV210)>>10 ) & 0x3FF;
		y[3] = ((*pV210) >> 10) & 0x3FF;
		//*(pU + 2)  = ( (*pV210)>>20 ) & 0x3FF;
		u[2] = ((*pV210) >> 20) & 0x3FF;


		//*(pY + 4)  = ( (*pV210)>>32 ) & 0x3FF;
		y[4] = ((*pV210) >> 32) & 0x3FF;
		//*(pV + 2)  = ( (*pV210)>>42 ) & 0x3FF;
		v[2] = ((*pV210) >> 42) & 0x3FF;
		//*(pY + 5)  = ( (*pV210)>>52 ) & 0x3FF;
		y[5] = ((*pV210) >> 52) & 0x3FF;

		pV210 = pV210 + 1;


		_int8 iIndex = 0;

		for (int i = 0; i < 3; i++)
		{
			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;
			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_2020(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;


			r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(y[iIndex], u[i], v[i]) >> 2);
			g[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(y[iIndex], u[i], v[i]) >> 2);
			b[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(y[iIndex], u[i], v[i]) >> 2);

			iIndex++;
			r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(y[iIndex], u[i], v[i]) >> 2);
			g[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(y[iIndex], u[i], v[i]) >> 2);
			b[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_2020_FullRange_CPU(y[iIndex], u[i], v[i]) >> 2);

			iIndex++;


			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;
			//r[iIndex] = (GET_10Bit_R_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//g[iIndex] = (GET_10Bit_G_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);
			//b[iIndex] = (GET_10Bit_B_FROM_10Bit_YUV_709(y[iIndex], u[i], v[i]) >> 2);

			//iIndex++;

		}

		(*output_int8_RGB) = r[0];
		*(output_int8_RGB + 1) = g[0];
		*(output_int8_RGB + 2) = b[0];
		*(output_int8_RGB + 3) = r[1];
		*(output_int8_RGB + 4) = g[1];
		*(output_int8_RGB + 5) = b[1];
		*(output_int8_RGB + 6) = r[2];
		*(output_int8_RGB + 7) = g[2];
		*(output_int8_RGB + 8) = b[2];
		*(output_int8_RGB + 9) = r[3];
		*(output_int8_RGB + 10) = g[3];
		*(output_int8_RGB + 11) = b[3];
		*(output_int8_RGB + 12) = r[4];
		*(output_int8_RGB + 13) = g[4];
		*(output_int8_RGB + 14) = b[4];
		*(output_int8_RGB + 15) = r[5];
		*(output_int8_RGB + 16) = g[5];
		*(output_int8_RGB + 17) = b[5];

		output_int8_RGB += 18;
	}
	}


	return 0;
}


int main()
{

	printf("Start:\n");


	cudaDeviceProp dProp;
	cudaGetDeviceProperties(&dProp, 0);

	int count;

	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);
	// sample2 v210->rgb888 test
	FILE *file;
	errno_t er = fopen_s(&file, "QuadHD_V210.yuv", "rb");
	if (er != 0)
	{
		return 0;
	}
	long lInputSize = 10240 * 2160;
	long lOutputSize = 3840 * 2160 * 3;

	unsigned char *uchFile = new unsigned char[lInputSize];

	unsigned char *i8Input = new unsigned char[lInputSize];
	unsigned char *i8Output = new unsigned char[lOutputSize];

	size_t rsize = fread_s(uchFile, lInputSize, 1, lInputSize, file);

	i8Input = (unsigned char *)uchFile;
	
	Timer t;

	t.reset();
	V210_2_RGB888_GPU( i8Input, lInputSize, i8Output, lOutputSize);
	//V210_2_RGB888_CPU(i8Input, lInputSize, i8Output, lOutputSize);

	printf("elapsed : %d\n", t.elapsed());
	
	FILE *wfile;
	er = fopen_s(&wfile, "QuadHD_RGB888_one_frame.yuv", "wb");
	fwrite((unsigned char *)i8Output, 1, lOutputSize, wfile);
	fflush(wfile);
	fclose(wfile);


    return 0;
}
