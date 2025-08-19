/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <string>
#include "data_contexts.cuh"
#include "kernel.cuh"
#include "kernel_factory.cuh"
#include "kernel_pool.cuh"

KernelPool::KernelPool(LucasPRPData* parentData, Pool pool) {
	// A Pool is made of several Kernels (subProcesses). Those kernels are the best performers in their Family. 
	// As a first step, we instanciate all candidates for the Families that are part of the Pool. This is the job of the kernel factory.
	KernelFactory* factory = new KernelFactory;
	std::vector<Process*> cs;
	switch (pool) {
	case Pool::FFT_GMEM: 
		Name = "FFT_GMEM";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::GMEM, parentData->totalLength));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
		break;
	case Pool::FFT_1SMEM:
		Name = "FFT_1SMEM";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_FFT_SQUARE_IFFT_UNWEIGHT, parentData->totalLength));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
		break;
	case Pool::FFT_2SMEM:
		Name = "FFT_2SMEM";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_FFTCOL_TWIDDLE, parentData->lengths[0], parentData->lengths[1]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[1], parentData->lengths[0]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_IFFTCOL_UNWEIGHT, parentData->lengths[0], parentData->lengths[1]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
		break;
	case Pool::FFT_2SMEM_TRANSPOSE:
		Name = "FFT_2SMEM_TRANSPOSE";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_TRANSPOSE, parentData->lengths[0], 1, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW, parentData->lengths[0], parentData->lengths[1]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_TWIDDLE, parentData->lengths[0], 1, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[1], parentData->lengths[0]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_TRANSPOSE, parentData->lengths[0], 1, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::IFFTROW, parentData->lengths[0], parentData->lengths[1]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_UNWEIGHT, parentData->lengths[0], 1, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
		break;
	case Pool::FFT_2SMEM_STORE_TRANSPOSED:
		Name = "FFT_2SMEM_STORE_TRANSPOSED";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_FFTROW, parentData->lengths[0], parentData->lengths[1]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_TWIDDLE, parentData->lengths[0], 1, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[1], parentData->lengths[0]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_TRANSPOSE, parentData->lengths[0], 1, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::IFFTROW_UNWEIGHT, parentData->lengths[0], parentData->lengths[1]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE_TRANSPOSED, parentData->totalLength, parentData->lengths[1]->n, parentData->lengths[0]->n));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2_TRANSPOSED, parentData->totalLength, parentData->lengths[1]->n, parentData->lengths[0]->n));
		}
		break;
	case Pool::FFT_3SMEM:
		Name = "FFT_3SMEM";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_FFTCOL_TWIDDLE, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTCOL_TWIDDLE, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[2], parentData->lengths[0]->n * parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_IFFTCOL, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_IFFTCOL_UNWEIGHT, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
		break;
	case Pool::FFT_3SMEM_TRANSPOSE:
		Name = "FFT_3SMEM_TRANSPOSE";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_TRANSPOSE, parentData->lengths[0], 1, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_TWIDDLE, parentData->lengths[0], 1, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE, parentData->lengths[1], parentData->lengths[0]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_TWIDDLE, parentData->lengths[1], parentData->lengths[0]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[2], parentData->lengths[0]->n * parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_TRANSPOSE, parentData->lengths[1], parentData->lengths[0]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::IFFTROW, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_BACK, parentData->lengths[2], parentData->lengths[0]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_TRANSPOSE, parentData->lengths[0], 1, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::IFFTROW, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_UNWEIGHT, parentData->lengths[0], 1, parentData->lengths[1]->n * parentData->lengths[2]->n));

		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
	break;	
	case Pool::FFT_3SMEM_STORE_TRANSPOSED:
		Name = "FFT_3SMEM_STORE_TRANSPOSED";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_FFTROW, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_TWIDDLE, parentData->lengths[0], 1, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE, parentData->lengths[1], parentData->lengths[0]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_TWIDDLE, parentData->lengths[1], parentData->lengths[0]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[2], parentData->lengths[0]->n * parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_TRANSPOSE, parentData->lengths[1], parentData->lengths[0]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::IFFTROW, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n, parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TRANSPOSE_BACK, parentData->lengths[2], parentData->lengths[0]->n, parentData->lengths[1]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_TRANSPOSE, parentData->lengths[0], 1, parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::IFFTROW_UNWEIGHT, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[1]->n * parentData->lengths[2]->n));

		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE_TRANSPOSED, parentData->totalLength, parentData->lengths[1]->n * parentData->lengths[2]->n, parentData->lengths[0]->n));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2_TRANSPOSED, parentData->totalLength, parentData->lengths[1]->n* parentData->lengths[2]->n, parentData->lengths[0]->n));
		}
		break;
	case Pool::FFT_4SMEM:
		Name = "FFT_4SMEM";
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::WEIGHT_FFTCOL_TWIDDLE, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n * parentData->lengths[3]->n, parentData->lengths[1]->n * parentData->lengths[2]->n * parentData->lengths[3]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTCOL_TWIDDLE, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n * parentData->lengths[3]->n, parentData->lengths[2]->n * parentData->lengths[3]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTCOL_TWIDDLE, parentData->lengths[2], parentData->lengths[0]->n * parentData->lengths[1]->n * parentData->lengths[3]->n, parentData->lengths[3]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::FFTROW_SQUARE_IFFTROW, parentData->lengths[3], parentData->lengths[0]->n * parentData->lengths[1]->n * parentData->lengths[2]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_IFFTCOL, parentData->lengths[2], parentData->lengths[0]->n * parentData->lengths[1]->n * parentData->lengths[3]->n, parentData->lengths[3]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_IFFTCOL, parentData->lengths[1], parentData->lengths[0]->n * parentData->lengths[2]->n * parentData->lengths[3]->n, parentData->lengths[2]->n * parentData->lengths[3]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::TWIDDLE_IFFTCOL_UNWEIGHT, parentData->lengths[0], parentData->lengths[1]->n * parentData->lengths[2]->n * parentData->lengths[3]->n, parentData->lengths[1]->n * parentData->lengths[2]->n * parentData->lengths[3]->n));
		subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::REDUCE2BASE, parentData->totalLength));
		if (!parentData->PRP) {
			subProcessesWithCandidates.push_back(factory->GetCandidates(parentData, Family::MINUS2, parentData->totalLength));
		}
		break;
	}
}

