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
#include "../core/atomic.cuh"
#include "../core/transpose_oop.cuh"
#include "r2b__tr.cuh"

///
/// Reduce to variable base 
/// with carry propagation
/// in-place, using global memory, one thread per element 
///
__global__ void reduceToBase_transposed(uint64_t* x, const uint8_t* __restrict__ widths, uint32_t stride)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int maxId = gridDim.x* blockDim.x;

	uint8_t w0 = widths[tid];
	uint64_t mask = (((uint64_t)1) << w0) - 1;
	uint64_t a = 0;
	uint8_t w = 0;

	// Keep the w least significant bits and send the most significant bits to the neighbours
	// First, we need to do x[tid] = (x[tid] + a) & mask
	// But because of the carry propagation, each x[tid] may be overwritten by several threads
	// So use an atomic function to make sure we write the correct data
	uint64_t old = atomicAddAnd(&x[tid], a, mask);

	// Then do carry propagation of the addition
	uint64_t carry = (old + a) >> w0;
	while (carry) {
		tid += stride;
		if (tid >= maxId) {
			uint32_t col = tid & (stride - 1);
			tid = (col + 1 >= stride) ? 0 : col + 1;
		}
			
		w = widths[tid];
		mask = (((uint64_t)1) << w) - 1;

		old = atomicAddAnd(&x[tid], carry, mask); // meaning old = x[tid]; x[tid] = (x[tid] + carry) & mask;
		carry = (old + carry) >> w;
	}
}

r2b__tr::r2b__tr(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Kernel(lucasPRPData, length, count, stride) {
	Name = "r2b__tr";
}

bool r2b__tr::Initialize() {
	try {
		cudaMallocTracked((void**)&widths, lucasPRPData->totalLength->n * sizeof(uint64_t));
		cudaCheckErrors("cudaMalloc (local widths) failed!");
	}
	catch (...) {
		return false;
	}
	dim3 threads_tr(16, 16);
	dim3 blocks_tr(data->count / 16, data->stride / 16);
	transpose_outofplace<16, 1, TranspositionOperation::NO_MUL, uint8_t> <<< blocks_tr, threads_tr, 0, common::stream >>> (lucasPRPData->widths, widths, nullptr);
	cudaCheckErrors("cudaFree (local widths transpose) failed!");
	return Kernel::Initialize();
}

void r2b__tr::Finalize() {
	cudaFreeTracked(widths);
	cudaCheckErrors("cudaFree (local widths) failed!");
	Kernel::Finalize();
}

void r2b__tr::Run() {
	/*  Reduce to irrational base with carry propagation */
	reduceToBase_transposed <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, widths, data->stride);
	cudaCheckErrors("Kernel launch failed: reduceToBase_Transposed");
}
