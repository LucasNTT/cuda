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
#include "fftc_tw__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void fftcol_twiddle__radix2_8to4k(uint64_t* xinput, uint32_t stride, const uint64_t* __restrict__ roots, const uint64_t* __restrict__ twiddleFactors) {
	__shared__ uint64_t s[1 << log_n];
	__shared__ uint64_t w[1 << (log_n - 1)];

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x * stride + blockIdx.x; // non-coalesced access
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
	int offset = (blockIdx.y * stride) << log_n;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] = xinput[tid + offset + i * gap * stride];
	}
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		w[threadIdx.x + i * gap] = roots[threadIdx.x + i * gap];
	}
	__syncthreads();

	// 2. Compute the small FFT in shared mem
	fft_radix2_pease_inplace<log_n, bfliesPerThread, false>(s, w);
	__syncthreads();

	// 3. Transfer from shared memory to global memory
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		xinput[tid + offset + i * gap * stride] = mul_Mod(s[threadIdx.x + i * gap], twiddleFactors[tid + i * gap * stride]);
	}
}

fftc_tw__r2_8to4k::fftc_tw__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: NttFactor(lucasPRPData, length, count, stride) {
	Name = "fftc_tw__r2_8to4k";
}

bool fftc_tw__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	return NttFactor::Initialize();
}

void fftc_tw__r2_8to4k::Run() {
	/*  weight, fft by column, twiddle twiddleFactors (all in shared memory) */
	dim3 dimGrid(data->stride, data->count / data->stride, 1);
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: fftcol_twiddle__radix2_8to4k<3, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 302: fftcol_twiddle__radix2_8to4k<3, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 304: fftcol_twiddle__radix2_8to4k<3, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 308: fftcol_twiddle__radix2_8to4k<3, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 401: fftcol_twiddle__radix2_8to4k<4, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 402: fftcol_twiddle__radix2_8to4k<4, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 404: fftcol_twiddle__radix2_8to4k<4, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 408: fftcol_twiddle__radix2_8to4k<4, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 501: fftcol_twiddle__radix2_8to4k<5, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 502: fftcol_twiddle__radix2_8to4k<5, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 504: fftcol_twiddle__radix2_8to4k<5, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 508: fftcol_twiddle__radix2_8to4k<5, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 601: fftcol_twiddle__radix2_8to4k<6, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 602: fftcol_twiddle__radix2_8to4k<6, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 604: fftcol_twiddle__radix2_8to4k<6, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 608: fftcol_twiddle__radix2_8to4k<6, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 701: fftcol_twiddle__radix2_8to4k<7, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 702: fftcol_twiddle__radix2_8to4k<7, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 704: fftcol_twiddle__radix2_8to4k<7, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 708: fftcol_twiddle__radix2_8to4k<7, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 801: fftcol_twiddle__radix2_8to4k<8, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 802: fftcol_twiddle__radix2_8to4k<8, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 804: fftcol_twiddle__radix2_8to4k<8, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 808: fftcol_twiddle__radix2_8to4k<8, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 901: fftcol_twiddle__radix2_8to4k<9, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 902: fftcol_twiddle__radix2_8to4k<9, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 904: fftcol_twiddle__radix2_8to4k<9, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 908: fftcol_twiddle__radix2_8to4k<9, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1001: fftcol_twiddle__radix2_8to4k<10, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1002: fftcol_twiddle__radix2_8to4k<10, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1004: fftcol_twiddle__radix2_8to4k<10, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1008: fftcol_twiddle__radix2_8to4k<10, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1101: fftcol_twiddle__radix2_8to4k<11, 1> <<< dimGrid, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1102: fftcol_twiddle__radix2_8to4k<11, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1104: fftcol_twiddle__radix2_8to4k<11, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1108: fftcol_twiddle__radix2_8to4k<11, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1202: fftcol_twiddle__radix2_8to4k<12, 2> <<< dimGrid, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1204: fftcol_twiddle__radix2_8to4k<12, 4> <<< dimGrid, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	case 1208: fftcol_twiddle__radix2_8to4k<12, 8> <<< dimGrid, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, data->stride, h_roots[0], twiddleFactors); break;
	default: throw std::runtime_error("");
	}
	cudaCheckErrors("Kernel launch failed: fftcol_twiddle__radix2_8to4k");
}
