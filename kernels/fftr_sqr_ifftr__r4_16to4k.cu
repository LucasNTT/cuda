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
#include "../core/fft_r4.cuh"
#include "fftr_sqr_ifftr__r4_16to4k.cuh"

template<uint32_t log_n> __global__ void fftrow_square_ifftrow__radix4_16to4k(uint64_t* xinput, uint64_t** roots, uint64_t** invRoots) {
	__shared__ uint64_t s[1 << log_n];

	// 1. Transfer from global memory to shared memory
	int tid = threadIdx.x + blockIdx.x * blockDim.x * 4;
	s[threadIdx.x] = xinput[tid];
	s[threadIdx.x + (1 << (log_n - 2))] = xinput[tid + (1 << (log_n - 2))];
	s[threadIdx.x + (1 << (log_n - 1))] = xinput[tid + (1 << (log_n - 1))];
	s[threadIdx.x + 3 * (1 << (log_n - 2))] = xinput[tid + 3 * (1 << (log_n - 2))];
	__syncthreads();

	// 2. Compute the small FFT in shared mem, and square
	fft_radix4_pease_inplace<log_n>(s, roots[0], roots[1], roots[2]);
	__syncthreads();

	ifft_radix4_pease_inplace<log_n>(s, invRoots[0], invRoots[1], invRoots[2]); // inverse fft
	__syncthreads();

	// 3. Transfer from shared memory to global memory
	xinput[tid] = s[threadIdx.x];
	xinput[tid + (1 << (log_n - 2))] = s[threadIdx.x + (1 << (log_n - 2))];
	xinput[tid + (1 << (log_n - 1))] = s[threadIdx.x + (1 << (log_n - 1))];
	xinput[tid + 3 * (1 << (log_n - 2))] = s[threadIdx.x + 3 * (1 << (log_n - 2))];
}

fftr_sqr_ifftr__r4_16to4k::fftr_sqr_ifftr__r4_16to4k(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Ntt(lucasPRPData, length, count, stride) {
	Name = "fftr_sqr_ifftr__r4_16to4k";
	radix = 4;
	log_radix = 2;
	nbRootBuffers = 3;
}

bool fftr_sqr_ifftr__r4_16to4k::Initialize() {
	if (data->length->log3_n != 0 || data->length->log5_n != 0 || data->length->log2_n < 4 || data->length->log2_n > 12 || data->length->log2_n % 2 == 1)
		return false;
	void (*kernel_ptr)(uint64_t * xinput, uint64_t * *roots, uint64_t * *invRoots);
	switch (data->length->log2_n) {
	case 4: kernel_ptr = fftrow_square_ifftrow__radix4_16to4k<4>; break;
	case 6: kernel_ptr = fftrow_square_ifftrow__radix4_16to4k<6>; break;
	case 8: kernel_ptr = fftrow_square_ifftrow__radix4_16to4k<8>; break;
	case 10:kernel_ptr = fftrow_square_ifftrow__radix4_16to4k<10>; break;
	case 12:kernel_ptr = fftrow_square_ifftrow__radix4_16to4k<12>; break;
	}
	uint32_t maxThreadsPerBlock = GetMaxThreadsPerBlock((const void*)kernel_ptr);

	if (data->length->n >> log_radix > maxThreadsPerBlock) {
		Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Not enough resources to run this kernel (%s)", Name.c_str());
		return false;
	}
	return Ntt::Initialize();
}

void fftr_sqr_ifftr__r4_16to4k::Run() {
	switch (data->length->log2_n) {
	case 4: fftrow_square_ifftrow__radix4_16to4k<4> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, roots, invRoots); break;
	case 6: fftrow_square_ifftrow__radix4_16to4k<6> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, roots, invRoots); break;
	case 8: fftrow_square_ifftrow__radix4_16to4k<8> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, roots, invRoots); break;
	case 10: fftrow_square_ifftrow__radix4_16to4k<10> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, roots, invRoots); break;
	case 12: fftrow_square_ifftrow__radix4_16to4k<12> <<< data->count, data->length->n / 4, 0, common::stream >>> (lucasPRPData->x, roots, invRoots); break;
	}
	cudaCheckErrors("Kernel launch failed: fftrow_square_ifftrow__radix4_16to4k");
}
