/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include "../data_contexts.cuh"
#include "../kernel.cuh" 
#include "../common/common.cuh"
#include "../common/arithmetics.cuh"
#include "../core/fft_r2.cuh"
#include "fftr_sqr_ifftr__cg_r2_8to4k.cuh"

using namespace cooperative_groups;

template<uint32_t log_n, uint32_t bfliesPerThread> __global__ void fftrow_square_ifftrow__radix2_8to4k(uint64_t* xinput, const uint64_t* __restrict__ roots, const uint64_t* __restrict__ invRoots) {
	extern __shared__ uint64_t s[];
	uint64_t* s_roots = (uint64_t*)(&s[1 << log_n]);
	thread_block block = this_thread_block();

	// 1. Transfer from global memory to shared memory
	memcpy_async(block, s, xinput + blockIdx.x * bfliesPerThread * blockDim.x * 2, bfliesPerThread * blockDim.x * 2 * sizeof(uint64_t));
	memcpy_async(block, s_roots, roots, bfliesPerThread * blockDim.x * sizeof(uint64_t));

	wait(block);
	block.sync();

	// 2. Compute the smalls FFT in shared mem, and square
	fft_radix2_pease_inplace<log_n, bfliesPerThread, true>(s, s_roots);
	block.sync();

	// 3. Inverse FFTs
	memcpy_async(block, s_roots, invRoots, bfliesPerThread * blockDim.x * sizeof(uint64_t));

	wait(block);
	block.sync();

	ifft_radix2_pease_inplace<log_n, bfliesPerThread>(s, s_roots); // inverse fft
	block.sync();

	// 4. Transfer from shared memory to global memory
	memcpy_async(block, xinput + blockIdx.x * bfliesPerThread * blockDim.x * 2, s, bfliesPerThread * blockDim.x * 2 * sizeof(uint64_t));
	wait(block);
}


fftr_sqr_ifftr__cg_r2_8to4k::fftr_sqr_ifftr__cg_r2_8to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t gap)
	: Ntt(lucasPRPData, length, count, gap) {
	Name = "fftr_sqr_ifftr__cg_r2_8to4k";
}

bool fftr_sqr_ifftr__cg_r2_8to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 3 || data->length->log2_n > 12)
		return false;
	return Ntt::Initialize();
}

void fftr_sqr_ifftr__cg_r2_8to4k::Run() {
	switch (100 * data->length->log2_n + bfliesPerThread) {
	case 301: fftrow_square_ifftrow__radix2_8to4k<3, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 302: fftrow_square_ifftrow__radix2_8to4k<3, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 304: fftrow_square_ifftrow__radix2_8to4k<3, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 308: fftrow_square_ifftrow__radix2_8to4k<3, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 401: fftrow_square_ifftrow__radix2_8to4k<4, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 402: fftrow_square_ifftrow__radix2_8to4k<4, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 404: fftrow_square_ifftrow__radix2_8to4k<4, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 408: fftrow_square_ifftrow__radix2_8to4k<4, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 501: fftrow_square_ifftrow__radix2_8to4k<5, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 502: fftrow_square_ifftrow__radix2_8to4k<5, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 504: fftrow_square_ifftrow__radix2_8to4k<5, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 508: fftrow_square_ifftrow__radix2_8to4k<5, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 601: fftrow_square_ifftrow__radix2_8to4k<6, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 602: fftrow_square_ifftrow__radix2_8to4k<6, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 604: fftrow_square_ifftrow__radix2_8to4k<6, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 608: fftrow_square_ifftrow__radix2_8to4k<6, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 701: fftrow_square_ifftrow__radix2_8to4k<7, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 702: fftrow_square_ifftrow__radix2_8to4k<7, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 704: fftrow_square_ifftrow__radix2_8to4k<7, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 708: fftrow_square_ifftrow__radix2_8to4k<7, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 801: fftrow_square_ifftrow__radix2_8to4k<8, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 802: fftrow_square_ifftrow__radix2_8to4k<8, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 804: fftrow_square_ifftrow__radix2_8to4k<8, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 808: fftrow_square_ifftrow__radix2_8to4k<8, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 901: fftrow_square_ifftrow__radix2_8to4k<9, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 902: fftrow_square_ifftrow__radix2_8to4k<9, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 904: fftrow_square_ifftrow__radix2_8to4k<9, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 908: fftrow_square_ifftrow__radix2_8to4k<9, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1001: fftrow_square_ifftrow__radix2_8to4k<10, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1002: fftrow_square_ifftrow__radix2_8to4k<10, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1004: fftrow_square_ifftrow__radix2_8to4k<10, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1008: fftrow_square_ifftrow__radix2_8to4k<10, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1101: fftrow_square_ifftrow__radix2_8to4k<11, 1> <<< data->count, data->length->n / 2, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1102: fftrow_square_ifftrow__radix2_8to4k<11, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1104: fftrow_square_ifftrow__radix2_8to4k<11, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1108: fftrow_square_ifftrow__radix2_8to4k<11, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1202: fftrow_square_ifftrow__radix2_8to4k<12, 2> <<< data->count, data->length->n / 4, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1204: fftrow_square_ifftrow__radix2_8to4k<12, 4> <<< data->count, data->length->n / 8, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	case 1208: fftrow_square_ifftrow__radix2_8to4k<12, 8> <<< data->count, data->length->n / 16, 3 * data->length->n / 2 * sizeof(uint64_t), common::stream >>> (lucasPRPData->x, h_roots[0], h_invRoots[0]); break;
	}
	cudaCheckErrors("Kernel launch failed: fftrow_square_ifftrow__radix2_8to4k");
}
