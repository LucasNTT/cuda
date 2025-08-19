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
#include "w_fft_sqr_ifft_uw__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void weight_fft_square_ifft_unweight__radix2_8to4k(uint64_t* xinput, const uint64_t* __restrict__ roots, const uint64_t* __restrict__ invRoots, const uint64_t* __restrict__ weights, const uint64_t* __restrict__ unweights)
{
	__shared__ uint64_t s[1 << log_n];
	__shared__ uint64_t w[1 << (log_n - 1)];

	// 1. Transfer from global memory to shared memory
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] = mul_Mod(xinput[threadIdx.x + i * gap], weights[threadIdx.x + i * gap]);
	}
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		w[threadIdx.x + i * gap] = roots[threadIdx.x + i * gap];
	}
	__syncthreads();

	// 2. Compute the small FFT in shared mem, and square
	fft_radix2_pease_inplace<log_n, bfliesPerThread, true>(s, w);

	// 3. Inverse FFT
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		w[threadIdx.x + i * gap] = invRoots[threadIdx.x + i * gap]; // load inverse roots
	}
	__syncthreads();

	ifft_radix2_pease_inplace<log_n, bfliesPerThread>(s, w); // inverse fft
	__syncthreads();

	// 4. Transfer from shared memory to global memory
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		xinput[threadIdx.x + i * gap] = mul_Mod(s[threadIdx.x + i * gap], unweights[threadIdx.x + i * gap]);
	}
}

w_fft_sqr_ifft_uw__r2_8to4k::w_fft_sqr_ifft_uw__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "w_fft_sqr_ifft_uw__r2_8to4k";
}

bool w_fft_sqr_ifft_uw__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	return Ntt::Initialize();
}

void w_fft_sqr_ifft_uw__r2_8to4k::Run() {
	/*  steps 1, 2, 3, 4, 5: weight, fft, square, ifft, unweight (all in shared memory) */
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: weight_fft_square_ifft_unweight__radix2_8to4k<3, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 302: weight_fft_square_ifft_unweight__radix2_8to4k<3, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 304: weight_fft_square_ifft_unweight__radix2_8to4k<3, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 308: weight_fft_square_ifft_unweight__radix2_8to4k<3, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 401: weight_fft_square_ifft_unweight__radix2_8to4k<4, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 402: weight_fft_square_ifft_unweight__radix2_8to4k<4, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 404: weight_fft_square_ifft_unweight__radix2_8to4k<4, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 408: weight_fft_square_ifft_unweight__radix2_8to4k<4, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 501: weight_fft_square_ifft_unweight__radix2_8to4k<5, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 502: weight_fft_square_ifft_unweight__radix2_8to4k<5, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 504: weight_fft_square_ifft_unweight__radix2_8to4k<5, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 508: weight_fft_square_ifft_unweight__radix2_8to4k<5, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 601: weight_fft_square_ifft_unweight__radix2_8to4k<6, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 602: weight_fft_square_ifft_unweight__radix2_8to4k<6, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 604: weight_fft_square_ifft_unweight__radix2_8to4k<6, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 608: weight_fft_square_ifft_unweight__radix2_8to4k<6, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 701: weight_fft_square_ifft_unweight__radix2_8to4k<7, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 702: weight_fft_square_ifft_unweight__radix2_8to4k<7, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 704: weight_fft_square_ifft_unweight__radix2_8to4k<7, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 708: weight_fft_square_ifft_unweight__radix2_8to4k<7, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 801: weight_fft_square_ifft_unweight__radix2_8to4k<8, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 802: weight_fft_square_ifft_unweight__radix2_8to4k<8, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 804: weight_fft_square_ifft_unweight__radix2_8to4k<8, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 808: weight_fft_square_ifft_unweight__radix2_8to4k<8, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 901: weight_fft_square_ifft_unweight__radix2_8to4k<9, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 902: weight_fft_square_ifft_unweight__radix2_8to4k<9, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 904: weight_fft_square_ifft_unweight__radix2_8to4k<9, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 908: weight_fft_square_ifft_unweight__radix2_8to4k<9, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1001: weight_fft_square_ifft_unweight__radix2_8to4k<10, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1002: weight_fft_square_ifft_unweight__radix2_8to4k<10, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1004: weight_fft_square_ifft_unweight__radix2_8to4k<10, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1008: weight_fft_square_ifft_unweight__radix2_8to4k<10, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1101: weight_fft_square_ifft_unweight__radix2_8to4k<11, 1> <<< 1, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1102: weight_fft_square_ifft_unweight__radix2_8to4k<11, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1104: weight_fft_square_ifft_unweight__radix2_8to4k<11, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1108: weight_fft_square_ifft_unweight__radix2_8to4k<11, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1202: weight_fft_square_ifft_unweight__radix2_8to4k<12, 2> <<< 1, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1204: weight_fft_square_ifft_unweight__radix2_8to4k<12, 4> <<< 1, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	case 1208: weight_fft_square_ifft_unweight__radix2_8to4k<12, 8> <<< 1, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0], lucasPRPData->weights, lucasPRPData->unweights); break;
	}
	cudaCheckErrors("Kernel launch failed: weight_fft_square_ifft_unweight__radix2_8to4k");
}
