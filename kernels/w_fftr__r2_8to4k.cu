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
#include "../common/memory_tracker.cuh"
#include "../core/fft_r2.cuh"
#include "../core/transpose_oop.cuh"
#include "w_fftr__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void weight_fftrow__radix2_8to4k(const uint64_t* __restrict__ xinput, uint64_t* xoutput, const uint64_t* __restrict__ roots, const uint64_t* __restrict__ weights) {
	__shared__ uint64_t s[1 << log_n];
	__shared__ uint64_t w[1 << (log_n - 1)];

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x + blockIdx.x * blockDim.x * 2 * bfliesPerThread;
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] = mul_Mod(xinput[tid + i * gap], weights[tid + i * gap]);
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
		xoutput[tid + i * gap] = s[threadIdx.x + i * gap];
	}
}

w_fftr__r2_8to4k::w_fftr__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "w_fftr__r2_8to4k";
}

bool w_fftr__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	try {
		cudaMallocTracked((void**)&weights, lucasPRPData->totalLength->n * sizeof(uint64_t));
		cudaCheckErrors("cudaMalloc (local weights) failed!");
	}
	catch (...) {
		return false;
	}
	dim3 threads_tr(16, 16);
	dim3 blocks_tr(data->count / 16, data->length->n / 16);
	transpose_outofplace<16, 1, TranspositionOperation::NO_MUL, uint64_t> <<< blocks_tr, threads_tr, 0, common::stream >>> (lucasPRPData->weights, weights, nullptr);
	cudaCheckErrors("cudaFree (local weights transpose) failed!");
	return Ntt::Initialize();
}

void w_fftr__r2_8to4k::Finalize() {
	Ntt::Finalize();
	cudaFreeTracked(weights);
	cudaCheckErrors("cudaFree (local weights) failed!");
}

void w_fftr__r2_8to4k::Run() {
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: weight_fftrow__radix2_8to4k<3, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 302: weight_fftrow__radix2_8to4k<3, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 304: weight_fftrow__radix2_8to4k<3, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 308: weight_fftrow__radix2_8to4k<3, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 401: weight_fftrow__radix2_8to4k<4, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 402: weight_fftrow__radix2_8to4k<4, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 404: weight_fftrow__radix2_8to4k<4, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 408: weight_fftrow__radix2_8to4k<4, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 501: weight_fftrow__radix2_8to4k<5, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 502: weight_fftrow__radix2_8to4k<5, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 504: weight_fftrow__radix2_8to4k<5, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 508: weight_fftrow__radix2_8to4k<5, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 601: weight_fftrow__radix2_8to4k<6, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 602: weight_fftrow__radix2_8to4k<6, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 604: weight_fftrow__radix2_8to4k<6, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 608: weight_fftrow__radix2_8to4k<6, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 701: weight_fftrow__radix2_8to4k<7, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 702: weight_fftrow__radix2_8to4k<7, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 704: weight_fftrow__radix2_8to4k<7, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 708: weight_fftrow__radix2_8to4k<7, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 801: weight_fftrow__radix2_8to4k<8, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 802: weight_fftrow__radix2_8to4k<8, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 804: weight_fftrow__radix2_8to4k<8, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 808: weight_fftrow__radix2_8to4k<8, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 901: weight_fftrow__radix2_8to4k<9, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 902: weight_fftrow__radix2_8to4k<9, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 904: weight_fftrow__radix2_8to4k<9, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 908: weight_fftrow__radix2_8to4k<9, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1001: weight_fftrow__radix2_8to4k<10, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1002: weight_fftrow__radix2_8to4k<10, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1004: weight_fftrow__radix2_8to4k<10, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1008: weight_fftrow__radix2_8to4k<10, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1101: weight_fftrow__radix2_8to4k<11, 1> <<< data->count,  data->length->n / 2, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1102: weight_fftrow__radix2_8to4k<11, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1104: weight_fftrow__radix2_8to4k<11, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1108: weight_fftrow__radix2_8to4k<11, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1202: weight_fftrow__radix2_8to4k<12, 2> <<< data->count,  data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1204: weight_fftrow__radix2_8to4k<12, 4> <<< data->count,  data->length->n / 8, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	case 1208: weight_fftrow__radix2_8to4k<12, 8> <<< data->count,  data->length->n / 16, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, h_roots[0], weights); break;
	}
	cudaCheckErrors("Kernel launch failed: weight_fftrow__radix2_8to4k");
}
