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


template<uint32_t log_n> inline __device__ void fft_radix4_pease_inplace(uint64_t* x, const uint64_t* __restrict__ w1, const uint64_t* __restrict__ w2, const uint64_t* __restrict__ w3) {
	int k = threadIdx.x << 2;
	uint64_t a, b, c, d, ac, bd, amc, bmd;
	//1st loop to (n-1)th loop
#pragma unroll	
	for (int j = 0; j <= (int)log_n - 4; j = j + 2) {
		a = x[threadIdx.x];
		b = x[threadIdx.x + (1 << (log_n - 2))];
		c = x[threadIdx.x + (1 << (log_n - 1))];
		d = x[threadIdx.x + 3 * (1 << (log_n - 2))];
		__syncthreads();
		ac = add_Mod(a, c); // now performs a small fft2^2 on [a, b, c, d]
		bd = add_Mod(b, d);
		amc = sub_Mod(a, c);
		bmd = shift48(sub_Mod(b, d)); // 2^{48} is a 4th root of unity (we know that 2^{192} = 1 mod p, so (2^{48})^4 = 1 mod p)
		x[k] = add_Mod(ac, bd);
		x[k + 1] = mul_Mod(sub_Mod(ac, bd), w2[threadIdx.x >> j << j]); // we swap elements 2 and 3 (bit-reversed fft)
		x[k + 2] = mul_Mod(add_Mod(amc, bmd), w1[threadIdx.x >> j << j]);
		x[k + 3] = mul_Mod(sub_Mod(amc, bmd), w3[threadIdx.x >> j << j]);
		__syncthreads();
	}

	//Last loop
	a = x[threadIdx.x];
	b = x[threadIdx.x + (1 << (log_n - 2))];
	c = x[threadIdx.x + (1 << (log_n - 1))];
	d = x[threadIdx.x + 3 * (1 << (log_n - 2))];
	__syncthreads();
	ac = add_Mod(a, c);
	bd = add_Mod(b, d);
	amc = sub_Mod(a, c);
	bmd = shift48(sub_Mod(b, d));
	x[k] = add_Mod(ac, bd);
	x[k + 1] = sub_Mod(ac, bd);
	x[k + 2] = add_Mod(amc, bmd);
	x[k + 3] = sub_Mod(amc, bmd);
	//square
	x[k] = mul_Mod(x[k], x[k]);
	x[k + 1] = mul_Mod(x[k + 1], x[k + 1]);
	x[k + 2] = mul_Mod(x[k + 2], x[k + 2]);
	x[k + 3] = mul_Mod(x[k + 3], x[k + 3]);
}

template<uint32_t log_n> inline __device__ void ifft_radix4_pease_inplace(uint64_t* x, const uint64_t* __restrict__ w1, const uint64_t* __restrict__ w2, const uint64_t* __restrict__ w3) {
	int k = threadIdx.x << 2;
	//1st loop
	uint64_t a = x[k];
	uint64_t b = x[k + 1];
	uint64_t c = x[k + 2];
	uint64_t d = x[k + 3];
	__syncthreads();
	uint64_t ab = add_Mod(a, b); // now performs a small ifft2^2 on [a, b, c, d]
	uint64_t cd = add_Mod(c, d);
	uint64_t amb = sub_Mod(a, b);
	uint64_t cmd = shift48(sub_Mod(d, c)); // Let w=2^{48} be the 4th root of unity. We want to find the inverse of w: 1/w = w^3 / w^4 = w^3 = -w. So we can use the shift48 but on the negate (we take sub_Mod(d, c) instead of sub_Mod(c, d)).
	x[threadIdx.x] = add_Mod(ab, cd);
	x[threadIdx.x + (1 << (log_n - 2))] = add_Mod(amb, cmd);
	x[threadIdx.x + (1 << (log_n - 1))] = sub_Mod(ab, cd);
	x[threadIdx.x + 3 * (1 << (log_n - 2))] = sub_Mod(amb, cmd);
	__syncthreads();

	//2nd loop to n-th loop
#pragma unroll	
	for (int j = (int)log_n - 4; j >= 0; j = j - 2) {
		a = x[k];
		b = mul_Mod(x[k + 1], w2[threadIdx.x >> j << j]);
		c = mul_Mod(x[k + 2], w1[threadIdx.x >> j << j]);
		d = mul_Mod(x[k + 3], w3[threadIdx.x >> j << j]);
		__syncthreads();
		ab = add_Mod(a, b);
		cd = add_Mod(c, d);
		amb = sub_Mod(a, b);
		cmd = shift48(sub_Mod(d, c));
		x[threadIdx.x] = add_Mod(ab, cd);
		x[threadIdx.x + (1 << (log_n - 2))] = add_Mod(amb, cmd);
		x[threadIdx.x + (1 << (log_n - 1))] = sub_Mod(ab, cd);
		x[threadIdx.x + 3 * (1 << (log_n - 2))] = sub_Mod(amb, cmd);
		__syncthreads();
	}
}

