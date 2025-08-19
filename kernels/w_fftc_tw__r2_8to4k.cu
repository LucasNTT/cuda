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
#include "w_fftc_tw__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void weight_fftcol_twiddle__radix2_8to4k(uint64_t* xinput, uint32_t stride, const uint64_t* __restrict__ roots, const uint64_t* __restrict__ weights, const uint64_t* __restrict__ twiddleFactors) {
	extern __shared__ uint64_t s[];
	uint64_t* s_roots = (uint64_t*)(&s[1 << log_n]);

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x * stride + blockIdx.x; // non-coalesced access		
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] = mul32_Mod(xinput[tid + i * gap * stride], weights[tid + i * gap * stride]);
	}
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		s_roots[threadIdx.x + i * gap] = roots[threadIdx.x + i * gap];
	}
	__syncthreads();

	// 2. Compute the smalls FFT in shared mem
	fft_radix2_pease_inplace<log_n, bfliesPerThread, false>(s, s_roots);
	__syncthreads();

	// 3. Transfer from shared memory to global memory
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		xinput[tid + i * gap * stride] = mul_Mod(s[threadIdx.x + i * gap], twiddleFactors[tid + i * gap * stride]);
	}
}


w_fftc_tw__r2_8to4k::w_fftc_tw__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t gap)
	: NttFactor(lucasPRPData, length, count, gap) {
	Name = "w_fftc_tw__r2_8to4k";
}

bool w_fftc_tw__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	return NttFactor::Initialize();
}

void w_fftc_tw__r2_8to4k::Run() {
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: weight_fftcol_twiddle__radix2_8to4k<3, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 302: weight_fftcol_twiddle__radix2_8to4k<3, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 304: weight_fftcol_twiddle__radix2_8to4k<3, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 308: weight_fftcol_twiddle__radix2_8to4k<3, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 401: weight_fftcol_twiddle__radix2_8to4k<4, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 402: weight_fftcol_twiddle__radix2_8to4k<4, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 404: weight_fftcol_twiddle__radix2_8to4k<4, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 408: weight_fftcol_twiddle__radix2_8to4k<4, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 501: weight_fftcol_twiddle__radix2_8to4k<5, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 502: weight_fftcol_twiddle__radix2_8to4k<5, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 504: weight_fftcol_twiddle__radix2_8to4k<5, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 508: weight_fftcol_twiddle__radix2_8to4k<5, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 601: weight_fftcol_twiddle__radix2_8to4k<6, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 602: weight_fftcol_twiddle__radix2_8to4k<6, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 604: weight_fftcol_twiddle__radix2_8to4k<6, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 608: weight_fftcol_twiddle__radix2_8to4k<6, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 701: weight_fftcol_twiddle__radix2_8to4k<7, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 702: weight_fftcol_twiddle__radix2_8to4k<7, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 704: weight_fftcol_twiddle__radix2_8to4k<7, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 708: weight_fftcol_twiddle__radix2_8to4k<7, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 801: weight_fftcol_twiddle__radix2_8to4k<8, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 802: weight_fftcol_twiddle__radix2_8to4k<8, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 804: weight_fftcol_twiddle__radix2_8to4k<8, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 808: weight_fftcol_twiddle__radix2_8to4k<8, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 901: weight_fftcol_twiddle__radix2_8to4k<9, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 902: weight_fftcol_twiddle__radix2_8to4k<9, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 904: weight_fftcol_twiddle__radix2_8to4k<9, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 908: weight_fftcol_twiddle__radix2_8to4k<9, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1001: weight_fftcol_twiddle__radix2_8to4k<10, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1002: weight_fftcol_twiddle__radix2_8to4k<10, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1004: weight_fftcol_twiddle__radix2_8to4k<10, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1008: weight_fftcol_twiddle__radix2_8to4k<10, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1101: weight_fftcol_twiddle__radix2_8to4k<11, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1102: weight_fftcol_twiddle__radix2_8to4k<11, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1104: weight_fftcol_twiddle__radix2_8to4k<11, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1108: weight_fftcol_twiddle__radix2_8to4k<11, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1202: weight_fftcol_twiddle__radix2_8to4k<12, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1204: weight_fftcol_twiddle__radix2_8to4k<12, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	case 1208: weight_fftcol_twiddle__radix2_8to4k<12, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], lucasPRPData->weights, twiddleFactors); break;
	}
	cudaCheckErrors("Kernel launch failed: weight_fftcol_twiddle__radix2_8to4k");
}
