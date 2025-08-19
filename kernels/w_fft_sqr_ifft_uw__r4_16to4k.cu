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
#include "../core/fft_r4.cuh"
#include "w_fft_sqr_ifft_uw__r4_16to4k.cuh"


template<uint32_t log_n> __global__ void weight_fft_square_ifft_unweight__radix4_16to4k(uint64_t* xinput, uint64_t** roots, uint64_t** invRoots, const uint64_t* __restrict__ weights, const uint64_t* __restrict__ unweights)
{
	__shared__ uint64_t s[1 << log_n];

	// 1. Transfer from global memory to shared memory
	s[threadIdx.x] = mul_Mod(xinput[threadIdx.x], weights[threadIdx.x]);
	s[threadIdx.x + (1 << (log_n - 2))] = mul_Mod(xinput[threadIdx.x + (1 << (log_n - 2))], weights[threadIdx.x + (1 << (log_n - 2))]);
	s[threadIdx.x + (1 << (log_n - 1))] = mul_Mod(xinput[threadIdx.x + (1 << (log_n - 1))], weights[threadIdx.x + (1 << (log_n - 1))]);
	s[threadIdx.x + 3 * (1 << (log_n - 2))] = mul_Mod(xinput[threadIdx.x + 3 * (1 << (log_n - 2))], weights[threadIdx.x + 3 * (1 << (log_n - 2))]);
	__syncthreads(); 

	// 2. Compute the small FFT in shared mem, and square
	fft_radix4_pease_inplace<log_n>(s, roots[0], roots[1], roots[2]);
	__syncthreads();

	ifft_radix4_pease_inplace<log_n>(s, invRoots[0], invRoots[1], invRoots[2]); // inverse fft
	__syncthreads();

	// 3. Transfer from shared memory to global memory
	xinput[threadIdx.x] = mul_Mod(s[threadIdx.x], unweights[threadIdx.x]);
	xinput[threadIdx.x + (1 << (log_n - 2))] = mul_Mod(s[threadIdx.x + (1 << (log_n - 2))], unweights[threadIdx.x + (1 << (log_n - 2))]);
	xinput[threadIdx.x + (1 << (log_n - 1))] = mul_Mod(s[threadIdx.x + (1 << (log_n - 1))], unweights[threadIdx.x + (1 << (log_n - 1))]);
	xinput[threadIdx.x + 3 * (1 << (log_n - 2))] = mul_Mod(s[threadIdx.x + 3 * (1 << (log_n - 2))], unweights[threadIdx.x + 3 * (1 << (log_n - 2))]);
}

w_fft_sqr_ifft_uw__r4_16to4k::w_fft_sqr_ifft_uw__r4_16to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "w_fft_sqr_ifft_uw__r4_16to4k";
	radix = 4;
	log_radix = 2;
	nbRootBuffers = 3;
}

bool w_fft_sqr_ifft_uw__r4_16to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 4 || data->length->log2_n > 12 || data->length->log2_n % 2 == 1)
		return false;

	void (*kernel_ptr)(uint64_t * xinput, uint64_t * *roots, uint64_t * *invRoots, const uint64_t * __restrict__ weights, const uint64_t * __restrict__ unweights);
	switch (data->length->log2_n) {
	case 4: kernel_ptr = weight_fft_square_ifft_unweight__radix4_16to4k<4>; break;
	case 6: kernel_ptr = weight_fft_square_ifft_unweight__radix4_16to4k<6>; break;
	case 8: kernel_ptr = weight_fft_square_ifft_unweight__radix4_16to4k<8>; break;
	case 10:kernel_ptr = weight_fft_square_ifft_unweight__radix4_16to4k<10>; break;
	case 12:kernel_ptr = weight_fft_square_ifft_unweight__radix4_16to4k<12>; break;
	}
	uint32_t maxThreadsPerBlock = GetMaxThreadsPerBlock((const void*)kernel_ptr);

	if (data->length->n >> log_radix > maxThreadsPerBlock) {
		Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Not enough resources to run this kernel (%s)", Name.c_str());
		return false;
	}

	if (data->length->log2_n == 12) {
		PreferredThreadCount = 1024;
	}	
	return Ntt::Initialize();
}

void w_fft_sqr_ifft_uw__r4_16to4k::Run() {
	/*  steps 1, 2, 3, 4, 5: weight, fft radix 4, square, ifft radix 4, unweight (all in shared memory) */
	switch (data->length->log2_n) {
	case 4: weight_fft_square_ifft_unweight__radix4_16to4k<4> <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, roots, invRoots, lucasPRPData->weights, lucasPRPData->unweights); break;
	case 6: weight_fft_square_ifft_unweight__radix4_16to4k<6> <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, roots, invRoots, lucasPRPData->weights, lucasPRPData->unweights); break;
	case 8: weight_fft_square_ifft_unweight__radix4_16to4k<8> <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, roots, invRoots, lucasPRPData->weights, lucasPRPData->unweights); break;
	case 10: weight_fft_square_ifft_unweight__radix4_16to4k<10> <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, roots, invRoots, lucasPRPData->weights, lucasPRPData->unweights); break;
	case 12: weight_fft_square_ifft_unweight__radix4_16to4k<12> <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, roots, invRoots, lucasPRPData->weights, lucasPRPData->unweights); break;
	}
	cudaCheckErrors("Kernel launch failed: weight_fft_square_ifft_unweight__radix4_16to4k");
}
