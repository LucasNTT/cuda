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

#pragma once

template<uint32_t log_n, uint32_t bfliesPerThread, bool square> inline __device__ void fft_radix2_pease_inplace(uint64_t* x, const uint64_t* __restrict__ roots)
{
	const uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
	uint64_t u[bfliesPerThread], v[bfliesPerThread];
	int tid;

	for (int j = 0; j <= (int)log_n - 2; j++) {
		tid = threadIdx.x;
#pragma unroll	
		for (int i = 0; i < bfliesPerThread; i++) {
			u[i] = x[tid];
			v[i] = x[tid + (1 << (log_n - 1))];
			tid += gap;
		}
		__syncthreads();
		tid = threadIdx.x * 2;
#pragma unroll	
		for (int i = 0; i < bfliesPerThread; i++) {
			x[tid] = add_Mod(u[i], v[i]);
			x[tid + 1] = mul_Mod(sub_Mod(u[i], v[i]), roots[tid >> (j + 1) << j]);
			tid += gap * 2;
		}
		__syncthreads();
	}
	//Last loop with square
	tid = threadIdx.x;
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		u[i] = x[tid];
		v[i] = x[tid + (1 << (log_n - 1))];
		tid += gap;
	}
	__syncthreads();
	tid = threadIdx.x * 2;
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		if (square) {
			uint64_t uv = add_Mod(u[i], v[i]);
			x[tid] = mul_Mod(uv, uv); // Square
			uv = sub_Mod(u[i], v[i]);
			x[tid + 1] = mul_Mod(uv, uv); // Square
		}
		else {
			x[tid] = add_Mod(u[i], v[i]);
			x[tid + 1] = sub_Mod(u[i], v[i]);
		}
		tid += gap * 2;
	}
	__syncthreads();
}

template<uint32_t log_n, uint32_t bfliesPerThread> inline __device__ void ifft_radix2_pease_inplace(uint64_t* x, const uint64_t* __restrict__ invRoots)
{
	const uint32_t gap = (1 << log_n) / bfliesPerThread / 2;
	uint64_t u[bfliesPerThread], v[bfliesPerThread];
	int tid = threadIdx.x * 2;
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		u[i] = x[tid];
		v[i] = x[tid + 1];
		tid += gap * 2;
	}
	__syncthreads();
	tid = threadIdx.x;
#pragma unroll	
	for (int i = 0; i < bfliesPerThread; i++) {
		x[tid] = add_Mod(u[i], v[i]);
		x[tid + (1 << (log_n - 1))] = sub_Mod(u[i], v[i]);
		tid += gap;
	}
	__syncthreads();

	for (int j = (int)log_n - 2; j >= 0; j--) {
		tid = threadIdx.x * 2;
#pragma unroll	
		for (int i = 0; i < bfliesPerThread; i++) {
			u[i] = x[tid];
			v[i] = mul_Mod(x[tid + 1], invRoots[tid >> (j + 1) << j]);
			tid += gap * 2;
		}
		__syncthreads();
		tid = threadIdx.x;
#pragma unroll	
		for (int i = 0; i < bfliesPerThread; i++) {
			x[tid] = add_Mod(u[i], v[i]);
			x[tid + (1 << (log_n - 1))] = sub_Mod(u[i], v[i]);
			tid += gap;
		}
		__syncthreads();
	}
}
