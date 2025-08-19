/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <vector>
#include "process.cuh"
#include "kernel_factory.cuh"
#include "kernels/m2.cuh"
#include "kernels/m2__tr.cuh"
#include "kernels/gmem.cuh"
#include "kernels/r2b.cuh"
#include "kernels/r2b__tr.cuh"
#include "kernels/w_fft_sqr_ifft_uw__r2_8to4k.cuh"
#include "kernels/fftr_sqr_ifftr__cg_r2_8to4k.cuh"
#include "kernels/w_fftc_tw__r2_8to4k.cuh"
#include "kernels/fftr_sqr_ifftr__r2_8to4k.cuh"
#include "kernels/tw_ifftc_uw__r2_8to4k.cuh"
#include "kernels/fftc_tw__r2_8to4k.cuh"
#include "kernels/tw_ifftc__r2_8to4k.cuh"
#include "kernels/w_fft_sqr_ifft_uw__r4_16to4k.cuh"
#include "kernels/w_fft_sqr_ifft_uw__r8_4096.cuh"
#include "kernels/fftr_sqr_ifftr__r2_32x32_1024.cuh"
#include "kernels/fftr_sqr_ifftr__r4_16to4k.cuh"
#include "kernels/transpose.cuh"
#include "kernels/transpose__back.cuh"
#include "kernels/w_transpose.cuh"
#include "kernels/transpose_tw.cuh"
#include "kernels/tw_transpose.cuh"
#include "kernels/transpose_uw.cuh"
#include "kernels/fftr__r2_8to4k.cuh"
#include "kernels/fftr__cg_r2_8to4k.cuh"
#include "kernels/ifftr__r2_8to4k.cuh"
#include "kernels/ifftr__cg_r2_8to4k.cuh"
#include "kernels/fftr_tw__r2_8to4k.cuh"
#include "kernels/w_fftr__r2_8to4k.cuh"
#include "kernels/ifftr_uw__r2_8to4k.cuh"


std::vector<Process*> KernelFactory::GetCandidates(LucasPRPData* lucasPRPData, Family family, Length* length, uint32_t count, uint32_t stride)
{
	// Get Candidates for a specific family of kernels
	std::vector<Process*> result; 
	switch (family) {
	case Family::MINUS2:
		result.push_back(new m2(lucasPRPData, length, count, stride));
		break;
	case Family::GMEM:
		result.push_back(new gmem_cooley_tukey__r2_8to64m(lucasPRPData, length, count, stride));
		result.push_back(new gmem_pease__r2_8to64m(lucasPRPData, length, count, stride));
		break;
	case Family::REDUCE2BASE: 
		result.push_back(new r2b(lucasPRPData, length, count, stride)); 
		break;
	case Family::WEIGHT_FFT_SQUARE_IFFT_UNWEIGHT: 
		result.push_back(new w_fft_sqr_ifft_uw__r2_8to4k(lucasPRPData, length, count, stride)); 
		result.push_back(new w_fft_sqr_ifft_uw__r4_16to4k(lucasPRPData, length, count, stride));
		result.push_back(new w_fft_sqr_ifft_uw__r8_4096(lucasPRPData, length, count, stride));
		break;
	case Family::WEIGHT_FFTCOL_TWIDDLE: 
		result.push_back(new w_fftc_tw__r2_8to4k(lucasPRPData, length, count, stride));
		break;
	case Family::FFTROW_SQUARE_IFFTROW: 
		result.push_back(new fftr_sqr_ifftr__r2_8to4k(lucasPRPData, length, count, stride));
		result.push_back(new fftr_sqr_ifftr__cg_r2_8to4k(lucasPRPData, length, count, stride));
		result.push_back(new fftr_sqr_ifftr__r2_32x32_1024(lucasPRPData, length, count, stride));
		result.push_back(new fftr_sqr_ifftr__r4_16to4k(lucasPRPData, length, count, stride));
		break;
	case Family::TWIDDLE_IFFTCOL_UNWEIGHT: 
		result.push_back(new tw_ifftc_uw__r2_8to4k(lucasPRPData, length, count, stride)); 
		break;
	case Family::FFTCOL_TWIDDLE: 
		result.push_back(new fftc_tw__r2_8to4k(lucasPRPData, length, count, stride)); 
		break;
	case Family::TWIDDLE_IFFTCOL: 
		result.push_back(new tw_ifftc__r2_8to4k(lucasPRPData, length, count, stride)); 
		break;
	case Family::WEIGHT_TRANSPOSE:
		result.push_back(new w_transpose(lucasPRPData, length, count, stride));
		break;
	case Family::FFTROW:
		result.push_back(new fftr__r2_8to4k(lucasPRPData, length, count, stride));
		result.push_back(new fftr__cg_r2_8to4k(lucasPRPData, length, count, stride));
		break;
	case Family::IFFTROW:
		result.push_back(new ifftr__r2_8to4k(lucasPRPData, length, count, stride));
		result.push_back(new ifftr__cg_r2_8to4k(lucasPRPData, length, count, stride));
		break;
	case Family::TRANSPOSE:
		result.push_back(new transpose(lucasPRPData, length, count, stride));
		break;
	case Family::TRANSPOSE_BACK:
		result.push_back(new transpose__back(lucasPRPData, length, count, stride));
		break;
	case Family::TRANSPOSE_TWIDDLE:
		result.push_back(new transpose_tw(lucasPRPData, length, count, stride));
		break;
	case Family::TWIDDLE_TRANSPOSE:
		result.push_back(new tw_transpose(lucasPRPData, length, count, stride));
		break;
	case Family::TRANSPOSE_UNWEIGHT:
		result.push_back(new transpose_uw(lucasPRPData, length, count, stride));
		break;
	case Family::WEIGHT_FFTROW:
		result.push_back(new w_fftr__r2_8to4k(lucasPRPData, length, count, stride));
		break;
	case Family::IFFTROW_UNWEIGHT:
		result.push_back(new ifftr_uw__r2_8to4k(lucasPRPData, length, count, stride));
		break;
	case Family::FFTROW_TWIDDLE:
		result.push_back(new fftr_tw__r2_8to4k(lucasPRPData, length, count, stride));
	case Family::REDUCE2BASE_TRANSPOSED:
		result.push_back(new r2b__tr(lucasPRPData, length, count, stride));
		break;
	case Family::MINUS2_TRANSPOSED:
		result.push_back(new m2__tr(lucasPRPData, length, count, stride));
		break;
	}

	// Once the kernel candidate have been initialized, check if specific preferred kernels have been specified in the ini file
	bool found = false;
	for (auto process : result) {
		if (lucasPRPData->preferredKernels.find(";" + process->Name + ";") != std::string::npos)
			found = true;
	}
	// if this is the case, we keep only the preferred kernels and delete the others
	if (found) {
		for (int i = static_cast<int>(result.size()) - 1; i >= 0; --i) {
			if(lucasPRPData->preferredKernels.find(";" + result[i]->Name + ";") == std::string::npos) {
				delete result[i];
				result.erase(result.begin() + i);
			}
		}
	}
	return result;
}
