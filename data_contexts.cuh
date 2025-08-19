/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <stdint.h>
#include <string>
#include <vector>

#pragma once
struct Length {
	uint32_t n = 1;			///> The FFT length (for example 2^20)
	uint32_t log2_n = 0;
	uint32_t log3_n = 0;
	uint32_t log5_n = 0;
}; 

struct DataContext {

};

struct MainProcessData : DataContext {
	// main parameters for the calculation
	bool PRP; ///> PRP pseudo-primality test or Lucas-Lehmer primality test
	uint32_t exponent;  ///> The Mersenne exponent q to be tested, ie: we test the number M = 2^exponent - 1
	uint32_t iterations;///> The requested number of iterations. When not specified, we perform a full test, meaning that iter = exponent - 2 (for LL test)
	uint32_t fromIteration = 1;
	uint32_t currentIteration;
	// Handling CTRL-C & communication
	bool interrupted = false;
	bool wasInterrupted = false;
	bool dumpResidue = false;
	bool dumpStats = true;
	bool dumpEnvironment = false;
	// Dumping the results
	std::string dumpFilename;///> This is where to store the residue 
	std::string statFilename;///> This is where to store the statistics (execution times)
	std::string envFilename; ///> This is where to store the environment (CUDA Version, GPU Version, ...)
	// preferences
	std::string preferredPool;
	std::string preferredKernels;
};

struct LucasPRPData : DataContext {
	// Lengths
	std::vector<Length*> lengths; ///> The configuration of the different smaller FFTs that need to be performed in order to process the full FFT. 
	Length* totalLength; ///> The length of the full FFT
	// Memory Buffers 
	uint64_t* x;		///> Device buffer used to store the PRP/Lucas-Lehmer suite during calculation x(n+1) = x(n)^2 with minus 2 for LL
	uint64_t* weights;  ///> The weights used to perform an IBDWT, ie B^k where B is the primary N-root of two (B^N = 2) 
	uint64_t* unweights;///> The inverse of the weights used to perform an IBDWT, ie 1 / B^k
	uint8_t* widths;    ///> The widths of the variable-base IBDWT, in bits. Each element width may vary by 1 bit (for example, 19, 19, 20, 19, 19, 20, 19, ...)
	std::string preferredKernels;
	bool PRP;
};

enum class Pool {
	FFT_GMEM,
	FFT_1SMEM,
	FFT_2SMEM,
	FFT_2SMEM_TRANSPOSE,
	FFT_2SMEM_STORE_TRANSPOSED,
	FFT_3SMEM,
	FFT_3SMEM_TRANSPOSE,
	FFT_3SMEM_STORE_TRANSPOSED,
	FFT_4SMEM
};

enum class Family {
	MINUS2,
	GMEM,
	REDUCE2BASE,
	WEIGHT_FFT_SQUARE_IFFT_UNWEIGHT,
	WEIGHT_FFTCOL_TWIDDLE,
	FFTROW_SQUARE_IFFTROW,
	TWIDDLE_IFFTCOL_UNWEIGHT,
	FFTCOL_TWIDDLE,
	TWIDDLE_IFFTCOL,
	WEIGHT_TRANSPOSE,
	FFTROW,
	TRANSPOSE_TWIDDLE,
	TWIDDLE_TRANSPOSE,
	IFFTROW,
	TRANSPOSE_UNWEIGHT,
	WEIGHT_FFTROW,
	IFFTROW_UNWEIGHT,
	REDUCE2BASE_TRANSPOSED,
	MINUS2_TRANSPOSED,
	TRANSPOSE,
	TRANSPOSE_BACK,
	FFTROW_TWIDDLE
};

struct KernelData : DataContext {
	Length* length; ///> Length of the small FFT
	uint32_t count = 1; ///> Number of small FFTs we need to process
	uint32_t stride = 1; ///> This is the stride between each element of the small FFT
};
