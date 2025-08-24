/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include "arithmetics.cuh"
#include "pre_calculation.cuh"

///
/// Pre-calculation Kernels
///   Weights & unweights, widths, root & invRoot
///
__global__ void calc_weights_widths(uint32_t n, uint32_t exponent, uint64_t* weights, uint64_t* unweights, uint8_t* widths) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0) {
		weights[0] = 1;
		unweights[0] = inv_Mod(n);
	}
	else {
		uint32_t r = (exponent * tid) & (n - 1); // (exponent * tid) mod N -- note: exponent * tid will overflow on 32 bits, but no worries we are only interrested by the lowest bits
		uint64_t exp = ((MODULO - 1) / 192 / n) * 5ULL * (uint64_t)(n - r);
		weights[tid] = pow_Mod(7, exp);
		unweights[tid] = inv_Mod(mul_Mod(weights[tid], n));
	}
	widths[tid] = (uint8_t)(ceil(1.0 * exponent * (tid + 1) / n) - ceil(1.0 * exponent * tid / n));
}

// Not used
__global__ void calc_weights_widths_stride(uint32_t n, uint32_t exponent, uint32_t stride, uint64_t* weights, uint64_t* unweights, uint8_t* widths) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0) {
		weights[0] = 1;
		unweights[0] = inv_Mod(n);
	}
	else {
		uint32_t r = (exponent * tid) & (n - 1); // (exponent * tid) mod n
		uint64_t exp = ((MODULO - 1) / 192 / n) * 5ULL * (uint64_t)(n - r);
		weights[blockIdx.x + (threadIdx.x << stride)] = pow_Mod(7, exp);
		unweights[blockIdx.x + (threadIdx.x << stride)] = inv_Mod(mul_Mod(weights[tid], n));
	}
	widths[tid] = (uint8_t)(ceil(1.0 * exponent * (tid + 1) / n) - ceil(1.0 * exponent * tid / n));
}

__global__ void calc_roots(uint32_t n, uint64_t* roots, uint64_t* invRoots) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t rootOne = pow_Mod(554, (MODULO - 1) / n); // We prefer 554 as a primitive n root of unity, because it generates powers of two as roots of unity for small FFTs up to 64
	roots[tid] = pow_Mod(rootOne, tid);
	invRoots[tid] = inv_Mod(roots[tid]);
}

__global__ void calc_roots_pow(uint32_t n, uint64_t* roots, uint64_t* invRoots, uint32_t pow) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t rootOne = pow_Mod(554, (MODULO - 1) / n);
	roots[tid] = pow_Mod(rootOne, tid * pow);
	invRoots[tid] = inv_Mod(roots[tid]);
}


__global__ void calc_brev_factors(uint32_t n, uint32_t log_n, uint32_t stride, uint64_t* twiddleFactors, uint64_t* invTwiddleFactors) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = tid & (stride - 1);
	int j = __brev(tid / stride) >> (32 - log_n);
	uint64_t rootOne = pow_Mod(554, (MODULO - 1) / n);
	tid += blockIdx.y * gridDim.x * blockDim.x;
	twiddleFactors[tid] = pow_Mod(rootOne, i * j);
	invTwiddleFactors[tid] = inv_Mod(twiddleFactors[tid]);
}

