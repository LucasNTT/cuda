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
#include "w_fft_sqr_ifft_uw__r8_4096.cuh"


inline __device__ void fft2(uint64_t& x0, uint64_t& x1)
{
	uint64_t tmp = x0;
	x0 = add_Mod(tmp, x1);
	x1 = sub_Mod(tmp, x1);
}

inline __device__ void fft4(uint64_t& x0, uint64_t& x1, uint64_t& x2, uint64_t& x3)
{
	// Cooley-Tukey inplace DIF (Gentleman-Sande)
	// 2 x fft2 followed by 2 x fft2
	// 2^{48} is the 4th root of unity in Z/pZ
	fft2(x0, x2);
	fft2(x1, x3);
	x3 = shift48(x3);
	fft2(x0, x1);
	fft2(x2, x3);
}

inline __device__ void fft8(uint64_t* x)
{
	// 4 x FFT2 followed by 2 x FFT4. Twiddle factors only applied on elements 5, 6 and 7.
	// 2^{24} is the 8th root of unity in Z/pZ
	fft2(x[0], x[4]);
	fft2(x[1], x[5]);
	fft2(x[2], x[6]);
	fft2(x[3], x[7]);

	x[5] = shift24(x[5]);
	x[6] = shift48(x[6]);
	x[7] = shift72(x[7]);

	fft4(x[0], x[1], x[2], x[3]);
	fft4(x[4], x[5], x[6], x[7]);
}

inline __device__ void ifft4(uint64_t& x0, uint64_t& x1, uint64_t& x2, uint64_t& x3)
{
	// Cooley-Tukey inplace DIT
	// 2 x fft2 followed by 2 x fft2
	// 2^{48} is the 4th root of unity in Z/pZ
	fft2(x0, x1);
	fft2(x2, x3);
	x3 = neg_Mod(shift48(x3));
	fft2(x0, x2);
	fft2(x1, x3);
}

inline __device__ void ifft8(uint64_t* x)
{
	// 4 x FFT2 followed by 2 x FFT4. Twiddle factors only applied on elements 5, 6 and 7.
	// 2^{24} is the 8th root of unity in Z/pZ
	ifft4(x[0], x[1], x[2], x[3]);
	ifft4(x[4], x[5], x[6], x[7]);

	x[5] = neg_Mod(shift72(x[5]));
	x[6] = neg_Mod(shift48(x[6]));
	x[7] = neg_Mod(shift24(x[7]));

	fft2(x[0], x[4]);
	fft2(x[1], x[5]);
	fft2(x[2], x[6]);
	fft2(x[3], x[7]);
}


inline __device__ void fft_radix8_pease_inplace(uint64_t* x, uint64_t** roots) {
	// Radix-8 - 512 bflies - 4 steps - 512x8 = 4096
	// roots is a pointer to 7 x buffers of 512 uint64_t each
	uint64_t bfly[8];
	int k = threadIdx.x << 3;
#pragma unroll
	for (int step = 0; step < 3; step++) {
		// now perform a small fft8 on each butterfly. Each element of this small fft8 is taken every 512 elements
		for (int i = 0; i < 8; i++)
			bfly[i] = x[threadIdx.x + (i << 9)];

		fft8(bfly); // inner butterfly is bit-reversed after this fft8

		// copy back the butterfly to x, by applying the twiddle factors
		x[k] = bfly[0]; // no twiddle factor on the 1st element of the butterfly
		for (int i = 1; i < 8; i++) {
			// Explaination of the line below : 
			// we must multiply the inner butterfly by a twiddle factor, stored in roots. Because  the fft is bit-reversed, we reintroduce a bit-reversion on 3 bits to get the appropriate power: __brev(i) >> (32 - 3) 
			// and regarding (step * 3), the 3 comes from log_radix
			x[k + i] = mul_Mod(bfly[i], roots[(__brev(i) >> (32 - 3)) - 1][threadIdx.x >> (step * 3) << (step * 3)]);
		}
		__syncthreads();
	}

	// last step with square
	for (int i = 0; i < 8; i++)
		bfly[i] = x[threadIdx.x + (i << 9)];
	fft8(bfly);
	for (int i = 0; i < 8; i++) {
		x[k + i] = mul_Mod(bfly[i], bfly[i]);
	}
	__syncthreads();
}

inline __device__ void ifft_radix8_pease_inplace(uint64_t* x, uint64_t** invRoots) {
#pragma unroll
	for (int step = 3; step >= 0; step--) {
		// now perform a small ifft8 on each butterfly.
		uint64_t bfly[8];
		int k = threadIdx.x << 3;
		bfly[0] = x[k];
		for (int i = 1; i < 8; i++)
			// because the input following the forward fft is bit-reversed, we need to use a bit-reversion to make sure to multiply by the appropriate invRoot 
			bfly[i] = mul_Mod(x[k + i], invRoots[(__brev(i) >> (32 - 3)) - 1][threadIdx.x >> (step * 3) << (step * 3)]);

		ifft8(bfly); // inner butterfly is *not* bit-reversed after this ifft8

		for (int i = 0; i < 8; i++) {
			x[threadIdx.x + (i << 9)] = bfly[i];
		}
		__syncthreads();
	}
}

__global__ void weight_fft_square_ifft_unweight__radix8_4096(uint64_t* xinput, uint64_t** roots, uint64_t** invRoots, const uint64_t* __restrict__ weights, const uint64_t* __restrict__ unweights)
{
	// 1 single block, 512 threads processing 8 elements each
	__shared__ uint64_t s[4096]; // 32kb
#pragma unroll
	for (int i = 0; i < 8; i++) {
		s[threadIdx.x + 512 * i] = mul_Mod(xinput[threadIdx.x + 512 * i], weights[threadIdx.x + 512 * i]);
	}
	__syncthreads();

	fft_radix8_pease_inplace(s, roots); // with square
	__syncthreads();

	ifft_radix8_pease_inplace(s, invRoots);
	//test_fft8(s);
	__syncthreads();

#pragma unroll
	for (int i = 0; i < 8; i++) {
		xinput[threadIdx.x + 512 * i] = mul_Mod(s[threadIdx.x + 512 * i], unweights[threadIdx.x + 512 * i]);
	}
}


w_fft_sqr_ifft_uw__r8_4096::w_fft_sqr_ifft_uw__r8_4096(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "w_fft_sqr_ifft_uw__r8_4096";
	radix = 8;
	log_radix = 3;
	nbRootBuffers = radix - 1;
}

bool w_fft_sqr_ifft_uw__r8_4096::Initialize() {
	if (data->length->log2_n != 12)
		return false;
	void (*kernel_ptr)(uint64_t * xinput, uint64_t * *roots, uint64_t * *invRoots, const uint64_t * __restrict__ weights, const uint64_t * __restrict__ unweights) = weight_fft_square_ifft_unweight__radix8_4096;
	if (data->length->n >> log_radix > GetMaxThreadsPerBlock((const void*)kernel_ptr)) {
		Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Not enough resources to run this kernel (%s)", Name.c_str());
		return false;
	}

	PreferredThreadCount = 512;
	return Ntt::Initialize();
}

void w_fft_sqr_ifft_uw__r8_4096::Run() {
	/*  steps 1, 2, 3, 4, 5: weight, fft radix 8, square, ifft radix 8, unweight (all in shared memory) */
	weight_fft_square_ifft_unweight__radix8_4096 <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, roots, invRoots, lucasPRPData->weights, lucasPRPData->unweights);
	cudaCheckErrors("Kernel launch failed: weight_fft_square_ifft_unweight__radix8_4096");
}
