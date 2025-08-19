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
#include "m2.cuh"

///
/// Subtract 2
/// Assuming that the kernel is launched with only 1 thread
///
__global__ void minus_two(uint64_t* x, const uint8_t* __restrict__ widths, int n)
{
	uint64_t a = x[0];
	x[0] = a - 2;
	if (a < 2) // carry propagation needed
	{
		uint64_t mask = (((uint64_t)1) << widths[0]) - 1;
		x[0] = x[0] & mask;
		int id = 0;
		bool cf = true;
		while (cf) {
			id = (id == n) ? 0 : id + 1;
			mask = (((uint64_t)1) << widths[id]) - 1;
			a = x[id];
			x[id] = (a - 1) & mask;
			cf = a == 0;
		}
	}
}


m2::m2(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Kernel(lucasPRPData, length, count, stride) {
	Name = "m2";
}

void m2::Run() {
	/*  subtract 2 */
	minus_two <<< 1, 1, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->widths, lucasPRPData->totalLength->n - 1);
	cudaCheckErrors("Kernel launch failed: minus_two");
}
