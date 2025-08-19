/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../data_contexts.cuh"
#include "../kernel.cuh"
#include "../common/common.cuh"
#include "../common/arithmetics.cuh"
#include "fftr_sqr_ifftr__r2_32x32_1024.cuh"

__global__ void fftrow_square_ifftrow__radix2_32x32_1024(uint64_t* xinput, const uint64_t* __restrict__ twiddleFactors, const uint64_t* __restrict__ invTwiddleFactors) {
	__shared__ uint64_t s[33][33];
	int tid = threadIdx.x + blockIdx.x * blockDim.x * 2;
	int numFFT = threadIdx.x & 31;
	int numBfly = threadIdx.x >> 5;

	s[numBfly][numFFT] = xinput[tid];
	s[numBfly + 16][numFFT] = xinput[tid + 512];
	__syncthreads();

	// 32 x fft-5 by col
	uint64_t u = s[numBfly][numFFT];
	uint64_t v = s[numBfly + 16][numFFT];
	__syncthreads();
	s[numBfly * 2][numFFT] = add_Mod(u, v);
	s[numBfly * 2 + 1][numFFT] = shift(sub_Mod(u, v), 6 * numBfly);
	__syncthreads();
#pragma unroll	
	for (int j = 1; j <= 3; j++) {
		u = s[numBfly][numFFT];
		v = s[numBfly + 16][numFFT];
		__syncthreads();
		s[numBfly * 2][numFFT] = add_Mod(u, v);
		s[numBfly * 2 + 1][numFFT] = shift(sub_Mod(u, v), 6 * (numBfly >> j << j));
		__syncthreads();
	}
	u = s[numBfly][numFFT];
	v = s[numBfly + 16][numFFT];
	__syncthreads();
	s[numBfly * 2][numFFT] = mul_Mod(add_Mod(u, v), twiddleFactors[numFFT + numBfly * 2 * 32]); // Factor
	s[numBfly * 2 + 1][numFFT] = mul_Mod(sub_Mod(u, v), twiddleFactors[numFFT + (numBfly * 2 + 1) * 32]); // Factor
	__syncthreads();
	
	// 32 x fft-5 by row
	u = s[numFFT][numBfly];
	v = s[numFFT][numBfly + 16];
	__syncthreads();
	s[numFFT][numBfly * 2] = add_Mod(u, v);
	s[numFFT][numBfly * 2 + 1] = shift(sub_Mod(u, v), 6 * numBfly);
	__syncthreads();
#pragma unroll	
	for (int j = 1; j <= 3; j++) {
		u = s[numFFT][numBfly];
		v = s[numFFT][numBfly + 16];
		__syncthreads();
		s[numFFT][numBfly * 2] = add_Mod(u, v);
		s[numFFT][numBfly * 2 + 1] = shift(sub_Mod(u, v), 6 * (numBfly >> j << j));
		__syncthreads();
	}
	u = s[numFFT][numBfly];
	v = s[numFFT][numBfly + 16];
	__syncthreads();
	s[numFFT][numBfly * 2] = sqr_Mod(add_Mod(u, v)); // square
	s[numFFT][numBfly * 2 + 1] = sqr_Mod(sub_Mod(u, v));
	__syncthreads();

	// Now the backward transform
	// 32 x ifft-5 by row
	u = s[numFFT][numBfly * 2]; 
	v = s[numFFT][numBfly * 2 + 1];
	__syncthreads();
	s[numFFT][numBfly] = add_Mod(u, v);
	s[numFFT][numBfly + 16] = sub_Mod(u, v);
	__syncthreads();
#pragma unroll	
	for (int j = 3; j >= 1; j--) {
		u = s[numFFT][numBfly * 2];
		v = neg_Mod(shift(s[numFFT][numBfly * 2 + 1], 96 - 6 * (numBfly >> j << j)));
		__syncthreads();
		s[numFFT][numBfly] = add_Mod(u, v);
		s[numFFT][numBfly + 16] = sub_Mod(u, v);
		__syncthreads();
	}
	u = s[numFFT][numBfly * 2];
	v = neg_Mod(shift(s[numFFT][numBfly * 2 + 1], 96 - 6 * numBfly));
	__syncthreads();
	s[numFFT][numBfly] = mul_Mod(add_Mod(u, v), invTwiddleFactors[numFFT * 32 + numBfly]);
	s[numFFT][numBfly + 16] = mul_Mod(sub_Mod(u, v), invTwiddleFactors[numFFT * 32 + numBfly + 16]);
	__syncthreads();

	// Then 32 x ifft-5 by col
	u = s[numBfly * 2][numFFT];
	v = s[numBfly * 2 + 1][numFFT];
	__syncthreads();
	s[numBfly][numFFT] = add_Mod(u, v);
	s[numBfly + 16][numFFT] = sub_Mod(u, v);
	__syncthreads();
#pragma unroll	
	for (int j = 3; j >= 1; j--) {
		u = s[numBfly * 2][numFFT];
		v = neg_Mod(shift(s[numBfly * 2 + 1][numFFT], 96 - 6 * (numBfly >> j << j)));
		__syncthreads();
		s[numBfly][numFFT] = add_Mod(u, v);
		s[numBfly + 16][numFFT] = sub_Mod(u, v);
		__syncthreads();
	}
	u = s[numBfly * 2][numFFT];
	v = neg_Mod(shift(s[numBfly * 2 + 1][numFFT], 96 - 6 * numBfly));
	__syncthreads();
	s[numBfly][numFFT] = add_Mod(u, v);
	s[numBfly + 16][numFFT] = sub_Mod(u, v);
	__syncthreads();
	
	// Copy back the final results to global memory
	xinput[tid] = s[numBfly][numFFT];
	xinput[tid + 512] = s[numBfly + 16][numFFT];
}


fftr_sqr_ifftr__r2_32x32_1024::fftr_sqr_ifftr__r2_32x32_1024(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: NttFactor(lucasPRPData, length, count, stride) {
	Name = "fftr_sqr_ifftr__r2_32x32_1024";
	nbRootBuffers = 0; // no pre-calculation of the roots
}

bool fftr_sqr_ifftr__r2_32x32_1024::Initialize() {
	if (data->length->log2_n != 10)
		return false;
	void (*kernel_ptr)(uint64_t * xinput, const uint64_t * __restrict__ twiddleFactors, const uint64_t * __restrict__ invTwiddleFactors) = fftrow_square_ifftrow__radix2_32x32_1024;
	if (data->length->n / 2 > GetMaxThreadsPerBlock((const void*)kernel_ptr)) {
		Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Not enough resources to run this kernel (%s)", Name.c_str());
		return false;
	}
	data->length->n = 32; // temporary override data values to calculate small twiddle twiddleFactors
	data->length->log2_n = 5;
	data->stride = 32;
	uint32_t count = data->count;
	data->count = 32;
	if (NttFactor::Initialize()) {
		data->length->n = 1024; // and put back the initial values
		data->length->log2_n = 10;
		data->stride = 1;
		data->count = count;
		PreferredThreadCount = 512;
		common::InitThreadsAndBlocks(data->length->n, PreferredThreadCount, blocks, threads);
		common::InitThreadsAndBlocks(data->length->n / 2, PreferredThreadCount, nttBlocks, nttThreads);
		return true;
	}
	else
		return false;
}

void fftr_sqr_ifftr__r2_32x32_1024::Run() {
	/*  fft by row (decomposed as row-column again), square, ifft by row (decomposed as row-column again), all in shared memory */
	fftrow_square_ifftrow__radix2_32x32_1024 <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, twiddleFactors, invTwiddleFactors);
	cudaCheckErrors("Kernel launch failed: fftrow_square_ifftrow__radix2_32x32_1024");
}
