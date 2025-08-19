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
#include "gmem.cuh"


///
/// Kernel that performs the weighting (IBDWT)
/// This is a simple in-place pointwise multiplication x = x * y
/// 1 thread computes 1 element, so n threads is necessary
///
__global__ void weight(uint64_t* x, const uint64_t* __restrict__ y)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	x[tid] = mul_Mod(x[tid], y[tid]);
}

///
/// Square
/// This is a simple pointwise multiplication x = x^2 in-place
///
__global__ void square(uint64_t* x)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	x[tid] = mul_Mod(x[tid], x[tid]);
}

///
/// Decimation-In-Frequency FFT (Gentleman-Sande)
/// This is the Cuda implementation of a basic FFT using global memory and without any speed optimization.
/// The length of the FFT is a power of 2.
/// Each thread treats 1 butterfly, so the Kernel must be launched with a number of blocks and threads equals to the half of the FFT-length.
/// Input is ordered, but output is bit-scrambled (but we don't care because our Inverse FFT accepts a bit-scrambled input).
/// log_n is the number of steps to perform (thus FFT-length = 2^log_n)
/// stepNumber is between 1 and log_n
/// roots is an array (Read-Only) with the precalculated roots of unity : roots[0] = 1, roots[1] = w, roots[2]= w^2, etc.
///
__global__ void fft_CT(uint64_t* x, int stepNumber, int log_n, const uint64_t* __restrict__ roots)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int dist = 1 << (log_n - stepNumber);
	int idx = tid >> (log_n - stepNumber);
	int modIdx = tid & (dist - 1); // int modIdx = tid % dist;
	int Exp = modIdx << (stepNumber - 1);

	// Get the indices of the butterfly 
	int a = modIdx + (idx << (log_n - stepNumber + 1));
	int b = a + dist;

	uint64_t u = x[a];
	uint64_t v = x[b];
	x[a] = add_Mod(u, v); // U = U + V
	x[b] = mul_Mod(sub_Mod(u, v), roots[Exp]); // V = W * (U - V)
}

///
/// Decimation-In-Time FFT (Cooley-Tukey)
/// This is the Cuda implementation of a basic FFT without any speed optimization.
/// This is a reverse FFT, so we use the inverse roots of unity (this the only difference between a forward FFT and a reverse FFT, the factor 1/N being applied in the unweight kernel)
/// The length of the FFT is a power of 2.
/// Each thread treats 1 butterfly, so the Kernel must be launched with a number of blocks and threads equals to the half of the FFT-length.
/// Input is bit-scrambled, but output is ordered.
/// log_n is the number of steps to perform (thus FFT-length = 2^log_n)
/// stepNumber is between log_n and 1
/// roots is an array (Read-Only) with the precalculated inverse roots of unity : roots[0] = 1, roots[1] = 1/w, roots[2]= 1/w^2, etc.
///
__global__ void ifft_CT(uint64_t* x, int stepNumber, int log_n, const uint64_t* __restrict__ roots)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int dist = 1 << (stepNumber - 1);
	int idx = tid >> (stepNumber - 1);
	int modIdx = tid & (dist - 1); // int modIdx = tid % dist;
	int Exp = modIdx << (log_n - stepNumber);

	// Get the indices of the butterfly 
	int a = modIdx + (idx << stepNumber);
	int b = a + dist;

	uint64_t u = x[a];
	uint64_t v = mul_Mod(x[b], roots[Exp]);
	x[a] = add_Mod(u, v);  // U = U + V * W
	x[b] = sub_Mod(u, v);  // U = U - V * W
}

///
/// Decimation-In-Frequency Pease FFT
/// Out-of-Place, bit-reversed
/// input data are in x1 and x2 (two halves of x), output data are written in y
/// The length of the FFT is a power of 2.
/// Each thread treats 1 butterfly, so the Kernel must be launched with a number of blocks and threads equals to the half of the FFT-length.
/// Input x is ordered, and output y is bit-reversed
/// log_n is the number of steps to perform (thus FFT-length = 2^log_n)
/// This time stepNumber is between 1 and log_n
/// roots is an array (Read-Only) with the precalculated roots of unity : roots[0] = 1, roots[1] = w, roots[2]= w^2, etc.
///
__global__ void fft_Pease(const uint64_t* __restrict__ x1, const uint64_t* __restrict__ x2, uint64_t* y, uint32_t stepNumber, const uint64_t* __restrict__ roots)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int id = tid << 1;
	int Exp = (tid >> stepNumber) << stepNumber;
	uint64_t u = x1[tid];
	uint64_t v = x2[tid];
	y[id] = add_Mod(u, v); // U = U + V
	y[id + 1] = mul_Mod(sub_Mod(u, v), roots[Exp]); // V = W * (U - V)
}

///
/// Decimation-In-Time Pease FFT
/// Out-of-Place, bit-reversed
/// input data are in x1 and x2 (two halves of x), output data are written in y
/// This is a reverse FFT, so we use the inverse roots of unity (this the only difference between a forward FFT and a reverse FFT, the factor 1/N being applied in the unweight kernel)
/// The length of the FFT is a power of 2.
/// Each thread treats 1 butterfly, so the Kernel must be launched with a number of blocks and threads equals to the half of the FFT-length.
/// Input x is ordered, and output y is bit-reversed
/// log_n is the number of steps to perform (thus FFT-length = 2^log_n)
/// This time stepNumber is between log_n and 1
/// roots is an array (Read-Only) with the precalculated inverse roots of unity : roots[0] = 1, roots[1] = 1/w, roots[2]= 1/w^2, etc.
///
__global__ void ifft_Pease(const uint64_t* __restrict__ x, uint64_t* y1, uint64_t* y2, uint32_t stepNumber, const uint64_t* __restrict__ roots)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int id = tid << 1;
	int Exp = (tid >> stepNumber) << stepNumber;
	uint64_t u = x[id];
	uint64_t v = mul_Mod(x[id + 1], roots[Exp]);
	y1[tid] = add_Mod(u, v);  // U = U + V * W
	y2[tid] = sub_Mod(u, v);  // U = U - V * W
}


gmem::gmem(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "gmem (abstract)";
}

void gmem::Run() {
	/* step 1: weight the input */
	weight <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->weights);
	cudaCheckErrors("Kernel launch failed: weight");

	/* step 2, 3 & 4: forward transform, square, backward transform */
	fft_square_ifft();

	/*  step 5: unweight */
	weight <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->unweights);
	cudaCheckErrors("Kernel launch failed: unweight");
}

void gmem::EvaluatePerformance() {
	Kernel::EvaluatePerformance();
}

////////////////////////////////////////////////////////

gmem_cooley_tukey__r2_8to64m::gmem_cooley_tukey__r2_8to64m(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: gmem(lucasPRPData, length, count, stride) {
	Name = "gmem_cooley_tukey__r2_8to64m";
}

void gmem_cooley_tukey__r2_8to64m::fft_square_ifft() {
	for (uint32_t i = 1; i <= data->length->log2_n; i++) {
		fft_CT <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, i, data->length->log2_n, h_roots[0]);
		cudaCheckErrors("Kernel launch failed: fft_CT");
	}

	square <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x);
	cudaCheckErrors("Kernel launch failed: square");

	for (uint32_t i = 1; i <= data->length->log2_n; i++) {
		ifft_CT <<< nttBlocks, nttThreads, 0, common::stream >>> (lucasPRPData->x, i, data->length->log2_n, h_invRoots[0]);
		cudaCheckErrors("Kernel launch failed: ifft_CT");
	}
}

////////////////////////////////////////////////////////

gmem_pease__r2_8to64m::gmem_pease__r2_8to64m(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: gmem(lucasPRPData, length, count, stride) {
	Name = "gmem_pease__r2_8to64m";
}

bool gmem_pease__r2_8to64m::Initialize() {
	bool result = Ntt::Initialize();
	PreferredThreadCount = 256;
	common::InitThreadsAndBlocks(data->length->n / radix, PreferredThreadCount, nttBlocks, nttThreads);
	return result;
}

void gmem_pease__r2_8to64m::fft_square_ifft() {

	int pin = 0;
	uint64_t* x_half1, * x_half2, * y, * y_half1, * y_half2;

	for (uint32_t i = 0; i < data->length->log2_n; i++) {
		x_half1 = pin == 0 ? lucasPRPData->x : lucasPRPData->x + data->length->n;
		x_half2 = x_half1 + data->length->n / 2;
		y = pin == 0 ? lucasPRPData->x + data->length->n : lucasPRPData->x;
		fft_Pease <<< nttBlocks, nttThreads, 0, common::stream >>> (x_half1, x_half2, y, i, h_roots[0]);
		cudaCheckErrors("Kernel launch failed: fft_Pease");
		pin = 1 - pin; // swap double buffer
	}

	square <<< blocks, threads, 0, common::stream >>> (y); // y is equals to the first or second half of lucasPRPData->x, depending on data->length->log2_n
	cudaCheckErrors("Kernel launch failed: square");

	for (uint32_t i = 0; i < data->length->log2_n; i++) {
		y = pin == 0 ? lucasPRPData->x : lucasPRPData->x + data->length->n;
		y_half1 = pin == 0 ? lucasPRPData->x + data->length->n : lucasPRPData->x;
		y_half2 = y_half1 + data->length->n / 2;
		ifft_Pease <<< nttBlocks, nttThreads, 0, common::stream >>> (y, y_half1, y_half2, data->length->log2_n - i - 1, h_invRoots[0]);
		cudaCheckErrors("Kernel launch failed: ifft_Pease");
		pin = 1 - pin; // swap double buffer
	}
	// with 2 transforms, we have an even number of iterations, so the final result is stored in the first half of lucasPRPData->x and not the second half
}

