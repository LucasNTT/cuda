/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../data_contexts.cuh"
#include "../kernel.cuh"
#include "../common/common.cuh"
#include "../common/arithmetics.cuh"
#include "../core/fft_r2.cuh"
#include "tw_ifftc__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void twiddle_ifftcol_radix2_8to4k(uint64_t* xinput, uint32_t stride, const uint64_t* __restrict__ invRoots, const uint64_t* __restrict__ invTwiddleFactors) {
	__shared__ uint64_t s[1 << log_n];
	__shared__ uint64_t w[1 << (log_n - 1)];

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x * stride + blockIdx.x; // non-coalesced access
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
	int offset = (blockIdx.y * stride) << log_n;;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] = mul_Mod(xinput[tid + offset + i * gap * stride], invTwiddleFactors[tid + i * gap * stride]);
	}
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		w[threadIdx.x + i * gap] = invRoots[threadIdx.x + i * gap];
	}
	__syncthreads();

	// 2. Compute the small FFT in shared mem, and square
	ifft_radix2_pease_inplace<log_n, bfliesPerThread>(s, w);
	__syncthreads();

	// 3. Transfer from shared memory to global memory
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		xinput[tid + offset + i * gap * stride] = s[threadIdx.x + i * gap];
	}
}

tw_ifftc__r2_8to4k::tw_ifftc__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: NttFactor(lucasPRPData, length, count, stride) {
	Name = "tw_ifftc__r2_8to4k";
}

bool tw_ifftc__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	return NttFactor::Initialize();
}

void tw_ifftc__r2_8to4k::Run() {
	/*  ifft by column */
	dim3 dimGrid(data->stride, data->count / data->stride, 1);
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: twiddle_ifftcol_radix2_8to4k<3, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 302: twiddle_ifftcol_radix2_8to4k<3, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 304: twiddle_ifftcol_radix2_8to4k<3, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 308: twiddle_ifftcol_radix2_8to4k<3, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 401: twiddle_ifftcol_radix2_8to4k<4, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 402: twiddle_ifftcol_radix2_8to4k<4, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 404: twiddle_ifftcol_radix2_8to4k<4, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 408: twiddle_ifftcol_radix2_8to4k<4, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 501: twiddle_ifftcol_radix2_8to4k<5, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 502: twiddle_ifftcol_radix2_8to4k<5, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 504: twiddle_ifftcol_radix2_8to4k<5, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 508: twiddle_ifftcol_radix2_8to4k<5, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 601: twiddle_ifftcol_radix2_8to4k<6, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 602: twiddle_ifftcol_radix2_8to4k<6, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 604: twiddle_ifftcol_radix2_8to4k<6, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 608: twiddle_ifftcol_radix2_8to4k<6, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 701: twiddle_ifftcol_radix2_8to4k<7, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 702: twiddle_ifftcol_radix2_8to4k<7, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 704: twiddle_ifftcol_radix2_8to4k<7, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 708: twiddle_ifftcol_radix2_8to4k<7, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 801: twiddle_ifftcol_radix2_8to4k<8, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 802: twiddle_ifftcol_radix2_8to4k<8, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 804: twiddle_ifftcol_radix2_8to4k<8, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 808: twiddle_ifftcol_radix2_8to4k<8, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 901: twiddle_ifftcol_radix2_8to4k<9, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 902: twiddle_ifftcol_radix2_8to4k<9, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 904: twiddle_ifftcol_radix2_8to4k<9, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 908: twiddle_ifftcol_radix2_8to4k<9, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1001: twiddle_ifftcol_radix2_8to4k<10, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1002: twiddle_ifftcol_radix2_8to4k<10, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1004: twiddle_ifftcol_radix2_8to4k<10, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1008: twiddle_ifftcol_radix2_8to4k<10, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1101: twiddle_ifftcol_radix2_8to4k<11, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1102: twiddle_ifftcol_radix2_8to4k<11, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1104: twiddle_ifftcol_radix2_8to4k<11, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1108: twiddle_ifftcol_radix2_8to4k<11, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1202: twiddle_ifftcol_radix2_8to4k<12, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1204: twiddle_ifftcol_radix2_8to4k<12, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	case 1208: twiddle_ifftcol_radix2_8to4k<12, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_invRoots[0], invTwiddleFactors); break;
	}
	cudaCheckErrors("Kernel launch failed: twiddle_ifftcol_radix2_8to4k");
}
