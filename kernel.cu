/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "common/common.cuh"
#include "common/arithmetics.cuh"
#include "common/pre_calculation.cuh"
#include "common/memory_tracker.cuh"
#include "data_contexts.cuh"
#include "kernel.cuh"

__global__ void random(uint64_t* x, unsigned long seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curandState cState;
	curand_init(seed, tid, 0, &cState);
	float randf = curand_uniform(&cState);
	randf *= MODULO;
	x[tid] = (int)truncf(randf);
}

Kernel::Kernel(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride) {
	this->lucasPRPData = lucasPRPData;
	data = new KernelData;
	data->length = length;
	data->count = count;
	data->stride = stride;
}

Kernel::~Kernel() {
	delete data;
}

bool Kernel::Initialize() {
	common::InitThreadsAndBlocks(data->length->n, PreferredThreadCount, blocks, threads);
	return true;
}

void Kernel::EvaluatePerformance() {
	// Initialize Random Data for Test
	random <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, 12345);
	cudaCheckErrors("Kernel launch failed: random");
	Process::EvaluatePerformance();
}

struct cudaFuncAttributes Kernel::PrintFuncAttributes(const void* func) {
	struct cudaFuncAttributes funcAttrib;
	cudaFuncGetAttributes(&funcAttrib, func);
	cudaCheckErrors("Kernel launch failed: cudaFuncGetAttributes");
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "numRegs=%d", funcAttrib.numRegs);
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "localSizeBytes=%d", funcAttrib.localSizeBytes);
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "maxDynamicSharedSizeBytes=%d", funcAttrib.maxDynamicSharedSizeBytes);
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "maxThreadsPerBlock=%d", funcAttrib.maxThreadsPerBlock);
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "preferredShmemCarveout=%d", funcAttrib.preferredShmemCarveout);
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "sharedSizeBytes=%d", funcAttrib.sharedSizeBytes);
	return funcAttrib;
}

uint32_t Kernel::GetMaxThreadsPerBlock(const void* func) {
	struct cudaFuncAttributes funcAttrib = PrintFuncAttributes(func);
	return (uint32_t)funcAttrib.maxThreadsPerBlock;
}

uint32_t Kernel::GetMaxActiveBlocksPerMultiprocessor(const void* func, int blockSize, int dynamicSMemSize) {
	int maxActiveBlocks = 0;
	cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, dynamicSMemSize);
	if (error != cudaSuccess)
		return 0;
	return (uint32_t)maxActiveBlocks;
}


// ---------------------------------------------------------------------------------------------------------
Ntt::Ntt(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Kernel(lucasPRPData, length, count, stride) {
}

bool Ntt::Initialize() {
	if (data->length->log2_n == 10) {
		PreferredThreadCount = 512;
	}
	if (data->length->log2_n == 11) {
		PreferredThreadCount = 1024;
	}
	// note: if log2_n == 12, we don't rely on nttBlocks & nttThreads by increasing the number of butterflies per thread
	if (Kernel::Initialize()) {
		common::InitThreadsAndBlocks(data->length->n / radix, PreferredThreadCount, nttBlocks, nttThreads);
		if (nbRootBuffers > 0) {
			h_roots = (uint64_t**)calloc(nbRootBuffers, sizeof(uint64_t*));
			h_invRoots = (uint64_t**)calloc(nbRootBuffers, sizeof(uint64_t*));
			if (!h_roots || !h_invRoots) {
				Logger::getInstance().WriteLine(VerbosityLevel::Normal, "Cannot instanciate roots!");
				return false;
			}

			cudaMallocTracked((void**)&roots, nbRootBuffers * sizeof(uint64_t*));
			cudaCheckErrors("cudaMalloc (roots) failed!");
			cudaMallocTracked((void**)&invRoots, nbRootBuffers * sizeof(uint64_t*));
			cudaCheckErrors("cudaMalloc (invRoots) failed!");

			InitializeRoots();

			cudaMemcpy(roots, h_roots, nbRootBuffers * sizeof(uint64_t*), cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy (h_roots) failed!");
			cudaMemcpy(invRoots, h_invRoots, nbRootBuffers * sizeof(uint64_t*), cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy (h_roots) failed!");
		}
		return true;
	}
	else
		return false;
}

void Ntt::InitializeRoots() {
	for (int i = 0; i < nbRootBuffers; i++) {
		cudaMallocTracked((void**)&h_roots[i], data->length->n / radix * sizeof(uint64_t));
		cudaCheckErrors("cudaMalloc (roots) failed!");
		cudaMallocTracked((void**)&h_invRoots[i], data->length->n / radix * sizeof(uint64_t));
		cudaCheckErrors("cudaMalloc (invRoots) failed!");

		calc_roots_pow <<< nttBlocks, nttThreads, 0, common::stream >>> (data->length->n, h_roots[i], h_invRoots[i], i + 1);
		cudaCheckErrors("Kernel launch failed: calc_roots");
	}
}

void Ntt::Finalize() {
	if (nbRootBuffers > 0) {
		for (int i = nbRootBuffers - 1; i >= 0; i--) {
			cudaFreeTracked(h_invRoots[i]);
			cudaCheckErrors("cudaFree (invRoots) failed!");
			cudaFreeTracked(h_roots[i]);
			cudaCheckErrors("cudaFree (roots) failed!");
		}
		cudaFreeTracked(invRoots);
		cudaCheckErrors("cudaFree (invRoots) failed!");
		cudaFreeTracked(roots);
		cudaCheckErrors("cudaFree (roots) failed!");
		free(h_invRoots);
		free(h_roots);
	}
	Kernel::Finalize();
}

void Ntt::EvaluatePerformance() {
	if (radix != 2) { // as of today, processing multiple butterflies per thread is ony supported on Radix-2
		Kernel::EvaluatePerformance();
	}
	else {
		float bestPerf = MaxPerf;
		int bestBfliesPerThread = 0;

		for (int i = 0; i <= 3; i++) {
			bfliesPerThread = 1 << i;
			uint32_t threadsPerBlock = data->length->n / radix / bfliesPerThread;
			if (threadsPerBlock <= 1024) {
				try {
					Performance = MaxPerf;
					Kernel::EvaluatePerformance();
					if (Performance != MaxPerf) {
						Logger::getInstance().StorePerformance(common::format("%s (bflies=%d)", Name.c_str(), bfliesPerThread), Performance);
						Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Butterflies per thread = %d --> Performance = %gms", bfliesPerThread, Performance);
						if (Performance < bestPerf) {
							bestPerf = Performance;
							bestBfliesPerThread = bfliesPerThread;
						}
					}
				}
				catch (...) {
					Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Butterflies per thread = %d --> Not a valid configuration", bfliesPerThread);
				}
			}
		}
		if (bestPerf < MaxPerf) {
			Performance = bestPerf;
			bfliesPerThread = bestBfliesPerThread;
			Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Butterflies per thread = %d is the fastest configuration.", bfliesPerThread);
		}
	}
}

// ---------------------------------------------------------------------------------------------------------
NttFactor::NttFactor(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
}

bool NttFactor::Initialize() {
	if (Ntt::Initialize()) {
		if (initializeTwiddleFactors) {
			cudaMallocTracked((void**)&twiddleFactors, data->stride * data->length->n * sizeof(uint64_t)); // full size buffer
			cudaCheckErrors("cudaMalloc (twiddleFactors) failed!");
			cudaMallocTracked((void**)&invTwiddleFactors, data->stride * data->length->n * sizeof(uint64_t)); // full size buffer
			cudaCheckErrors("cudaMalloc (invTwiddleFactors) failed!");
			calc_brev_factors <<< data->stride * blocks, threads, 0, common::stream >>> (data->stride * data->length->n, data->length->log2_n, data->stride, twiddleFactors, invTwiddleFactors);
			cudaCheckErrors("Kernel launch failed: calc_brev_factors");
		}
		return true;
	}
	else
		return false;
}

void NttFactor::Finalize() {
	if (initializeTwiddleFactors) {
		cudaFreeTracked(invTwiddleFactors);
		cudaCheckErrors("cudaFree (invTwiddleFactors) failed!");
		cudaFreeTracked(twiddleFactors);
		cudaCheckErrors("cudaFree (twiddleFactors) failed!");
	}
	Ntt::Finalize();
}

// ---------------------------------------------------------------------------------------------------------
Transpose::Transpose(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Kernel(lucasPRPData, length, count, stride) {
}

bool Transpose::Initialize() {
	if (Kernel::Initialize()) {
		if (initializeTwiddleFactors) {
			cudaMallocTracked((void**)&twiddleFactors, data->count * data->stride * data->length->n * sizeof(uint64_t)); // full size buffer
			cudaCheckErrors("cudaMalloc (twiddleFactors) failed!");
			cudaMallocTracked((void**)&invTwiddleFactors, data->count * data->stride * data->length->n * sizeof(uint64_t)); // full size buffer
			cudaCheckErrors("cudaMalloc (invTwiddleFactors) failed!");
			dim3 blocks_f(data->stride * blocks, data->count);
			calc_brev_factors <<< blocks_f, threads, 0, common::stream >>> (data->stride * data->length->n, data->length->log2_n, data->stride, twiddleFactors, invTwiddleFactors);
			cudaCheckErrors("Kernel launch failed: calc_brev_factors");
		}
		return true;
	}
	else
		return false;
}

void Transpose::Finalize() {
	if (initializeTwiddleFactors) {
		cudaFreeTracked(invTwiddleFactors);
		cudaCheckErrors("cudaFree (invTwiddleFactors) failed!");
		cudaFreeTracked(twiddleFactors);
		cudaCheckErrors("cudaFree (twiddleFactors) failed!");
	}
	Kernel::Finalize();
}

void Transpose::EvaluatePerformance() {
	float bestPerf = MaxPerf;
	int bestTileSize = 4;
	int bestElementsPerThread = 1;
	for (int i = 2; i <= 5; i++) { // tileSize from 4 to 32
		for (int j = 0; j <= 4; j++) { // elementsPerThread from 1 to 16
			tileSize = 1 << i;
			elementsPerThread = 1 << j;
			dim3 threads(tileSize, tileSize / elementsPerThread);
			dim3 blocks(data->count / tileSize, data->length->n / tileSize);
			try {
				Performance = MaxPerf;
				Kernel::EvaluatePerformance();
				Logger::getInstance().StorePerformance(common::format("%s (tileSize=%02d, elementsPerThread=%02d)", Name.c_str(), tileSize, elementsPerThread), Performance);
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "tileSize=%d, elementsPerThread=%d --> Performance = %gms", tileSize, elementsPerThread, Performance);
				if (Performance < bestPerf) {
					bestPerf = Performance;
					bestTileSize = tileSize;
					bestElementsPerThread = elementsPerThread;
				}
			}
			catch (...) {
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "tileSize=%d, elementsPerThread=%d --> Not a valid configuration", tileSize, elementsPerThread);
			}
		}
	}
	Performance = bestPerf;
	tileSize = bestTileSize;
	elementsPerThread = bestElementsPerThread;
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "tileSize=%d, elementsPerThread=%d is the fastest configuration.", tileSize, elementsPerThread);
}

