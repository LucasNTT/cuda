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
#include "ifftr_uw__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void ifftrow_unweight__radix2_8to4k(const uint64_t* __restrict__ xinput, uint64_t* xoutput, const uint64_t* __restrict__ invRoots, const uint64_t* __restrict__ unweights) {
	__shared__ uint64_t s[1 << log_n];
	__shared__ uint64_t w[1 << (log_n - 1)];

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x + blockIdx.x * blockDim.x * 2 * bfliesPerThread;
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] =xinput[tid + i * gap];
	}
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		w[threadIdx.x + i * gap] = invRoots[threadIdx.x + i * gap];
	}
	__syncthreads();

	// 2. Compute the small inverse FFT in shared mem
	ifft_radix2_pease_inplace<log_n, bfliesPerThread>(s, w);
	__syncthreads();

	// 3. Transfer from shared memory to global memory
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		xoutput[tid + i * gap] = mul_Mod(s[threadIdx.x + i * gap], unweights[tid + i * gap]);
	}
}

ifftr_uw__r2_8to4k::ifftr_uw__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "ifftr_uw__r2_8to4k";
}

bool ifftr_uw__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	try {
		cudaMallocTracked((void**)&unweights, lucasPRPData->totalLength->n * sizeof(uint64_t));
		cudaCheckErrors("cudaMalloc (local unweights) failed!");
	}
	catch (...) {
		return false;
	}
	dim3 threads_tr(16, 16);
	dim3 blocks_tr(data->count / 16, data->length->n / 16);
	transpose_outofplace<16, 1, TranspositionOperation::NO_MUL, uint64_t> <<< blocks_tr, threads_tr, 0, common::stream >>> (lucasPRPData->unweights, unweights, nullptr);
	cudaCheckErrors("cudaFree (local unweights transpose) failed!");
	return Ntt::Initialize();
}

void ifftr_uw__r2_8to4k::Finalize() {
	cudaFreeTracked(unweights);
	cudaCheckErrors("cudaFree (local unweights) failed!");
	Ntt::Finalize();
}

void ifftr_uw__r2_8to4k::Run() {
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: ifftrow_unweight__radix2_8to4k<3, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 302: ifftrow_unweight__radix2_8to4k<3, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 304: ifftrow_unweight__radix2_8to4k<3, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 308: ifftrow_unweight__radix2_8to4k<3, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 401: ifftrow_unweight__radix2_8to4k<4, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 402: ifftrow_unweight__radix2_8to4k<4, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 404: ifftrow_unweight__radix2_8to4k<4, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 408: ifftrow_unweight__radix2_8to4k<4, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 501: ifftrow_unweight__radix2_8to4k<5, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 502: ifftrow_unweight__radix2_8to4k<5, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 504: ifftrow_unweight__radix2_8to4k<5, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 508: ifftrow_unweight__radix2_8to4k<5, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 601: ifftrow_unweight__radix2_8to4k<6, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 602: ifftrow_unweight__radix2_8to4k<6, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 604: ifftrow_unweight__radix2_8to4k<6, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 608: ifftrow_unweight__radix2_8to4k<6, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 701: ifftrow_unweight__radix2_8to4k<7, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 702: ifftrow_unweight__radix2_8to4k<7, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 704: ifftrow_unweight__radix2_8to4k<7, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 708: ifftrow_unweight__radix2_8to4k<7, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 801: ifftrow_unweight__radix2_8to4k<8, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 802: ifftrow_unweight__radix2_8to4k<8, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 804: ifftrow_unweight__radix2_8to4k<8, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 808: ifftrow_unweight__radix2_8to4k<8, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 901: ifftrow_unweight__radix2_8to4k<9, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 902: ifftrow_unweight__radix2_8to4k<9, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 904: ifftrow_unweight__radix2_8to4k<9, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 908: ifftrow_unweight__radix2_8to4k<9, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1001: ifftrow_unweight__radix2_8to4k<10, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1002: ifftrow_unweight__radix2_8to4k<10, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1004: ifftrow_unweight__radix2_8to4k<10, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1008: ifftrow_unweight__radix2_8to4k<10, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1101: ifftrow_unweight__radix2_8to4k<11, 1> <<< data->count, data->length->n / 2, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1102: ifftrow_unweight__radix2_8to4k<11, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1104: ifftrow_unweight__radix2_8to4k<11, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1108: ifftrow_unweight__radix2_8to4k<11, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1202: ifftrow_unweight__radix2_8to4k<12, 2> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1204: ifftrow_unweight__radix2_8to4k<12, 4> <<< data->count, data->length->n / 8, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	case 1208: ifftrow_unweight__radix2_8to4k<12, 8> <<< data->count, data->length->n / 16, 0, common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->x, h_invRoots[0], unweights); break;
	}
	cudaCheckErrors("Kernel launch failed: ifftrow_unweight__radix2_8to4k");
}
