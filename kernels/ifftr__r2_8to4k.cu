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
#include "ifftr__r2_8to4k.cuh"

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void ifftrow__radix2_8to4k(uint64_t* xinput, const uint64_t* __restrict__ invRoots) {
	extern __shared__ uint64_t s[];
	uint64_t* s_roots = (uint64_t*)(&s[1 << log_n]);

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x + blockIdx.x * blockDim.x * 2 * bfliesPerThread;
	constexpr uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		s[threadIdx.x + i * gap] = xinput[tid + i * gap];
	}
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		s_roots[threadIdx.x + i * gap] = invRoots[threadIdx.x + i * gap];
	}
	__syncthreads();

	// 2. Compute the smalls inverese FFT in shared mem
	ifft_radix2_pease_inplace<log_n, bfliesPerThread>(s, s_roots);
	__syncthreads();

	// 3. Transfer from shared memory to global memory
#pragma unroll	
	for (int i = 0; i < 2 * bfliesPerThread; i++) {
		xinput[tid + i * gap] = s[threadIdx.x + i * gap];
	}
}


ifftr__r2_8to4k::ifftr__r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t gap)
	: Ntt(lucasPRPData, length, count, gap) {
	Name = "ifftr__r2_8to4k";
}

bool ifftr__r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	return Ntt::Initialize();
}

void ifftr__r2_8to4k::Run() {
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: ifftrow__radix2_8to4k<3, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 302: ifftrow__radix2_8to4k<3, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 304: ifftrow__radix2_8to4k<3, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 308: ifftrow__radix2_8to4k<3, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 401: ifftrow__radix2_8to4k<4, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 402: ifftrow__radix2_8to4k<4, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 404: ifftrow__radix2_8to4k<4, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 408: ifftrow__radix2_8to4k<4, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 501: ifftrow__radix2_8to4k<5, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 502: ifftrow__radix2_8to4k<5, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 504: ifftrow__radix2_8to4k<5, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 508: ifftrow__radix2_8to4k<5, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 601: ifftrow__radix2_8to4k<6, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 602: ifftrow__radix2_8to4k<6, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 604: ifftrow__radix2_8to4k<6, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 608: ifftrow__radix2_8to4k<6, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 701: ifftrow__radix2_8to4k<7, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 702: ifftrow__radix2_8to4k<7, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 704: ifftrow__radix2_8to4k<7, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 708: ifftrow__radix2_8to4k<7, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 801: ifftrow__radix2_8to4k<8, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 802: ifftrow__radix2_8to4k<8, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 804: ifftrow__radix2_8to4k<8, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 808: ifftrow__radix2_8to4k<8, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 901: ifftrow__radix2_8to4k<9, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 902: ifftrow__radix2_8to4k<9, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 904: ifftrow__radix2_8to4k<9, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 908: ifftrow__radix2_8to4k<9, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1001: ifftrow__radix2_8to4k<10, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1002: ifftrow__radix2_8to4k<10, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1004: ifftrow__radix2_8to4k<10, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1008: ifftrow__radix2_8to4k<10, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1101: ifftrow__radix2_8to4k<11, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1102: ifftrow__radix2_8to4k<11, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1104: ifftrow__radix2_8to4k<11, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1108: ifftrow__radix2_8to4k<11, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1202: ifftrow__radix2_8to4k<12, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1204: ifftrow__radix2_8to4k<12, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	case 1208: ifftrow__radix2_8to4k<12, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x + lucasPRPData->totalLength->n, h_invRoots[0]); break;
	}
	cudaCheckErrors("Kernel launch failed: ifftrow__radix2_8to4k");
}
