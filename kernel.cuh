/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "process.cuh"
#include "data_contexts.cuh"

#pragma once
class Kernel : public Process {
public:
	Kernel(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride); // constructor
	~Kernel() override; // destructor
	virtual bool Initialize() override;
	int PreferredThreadCount = 256;
protected:
	LucasPRPData* lucasPRPData = NULL;
	KernelData* data = NULL;
	int blocks, threads;
	virtual void EvaluatePerformance() override;
	uint32_t GetMaxThreadsPerBlock(const void* func);
	uint32_t GetMaxActiveBlocksPerMultiprocessor(const void* func, int blockSize, int dynamicSMemSize);
	struct cudaFuncAttributes PrintFuncAttributes(const void* func);
};

class Ntt : public Kernel {
public:
	Ntt(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride); // constructor
	virtual bool Initialize() override;
	virtual void Finalize() override;
protected:
	int nttBlocks, nttThreads;
	int nbRootBuffers = 1; ///> The number of root buffers. For a regular radix-2 fft, we need one half-size buffer. For a radix-4 fft, we pre-calculate 3 quarter-size buffers. For a 4-step Bailey fft all done in smem (FFT_1SMEM), we need 2 buffers for the two small inner fft.
	uint32_t radix = 2; ///> classic radix-2 NTT involve n/2 threads processing n/2 butterflies of 2 elements, and require n/2 pre-calculated roots of unity
	uint32_t log_radix = 1; ///> 2^log_radix = radix
	uint64_t** roots; ///> The roots of unity (half-size buffer for radix 2) of the FFT, ie: w^k where k in [0, n/2 -1] and w is the primary N-root of unity (w^n = 1)
	uint64_t** invRoots; ///> The inverse roots of unity (half-size buffer) of the FFT, used for the backward transform, ie: 1 / w^k
	uint64_t** h_roots; ///> Copy of the above pointers in the host 
	uint64_t** h_invRoots; ///> Copy of the above pointers in the host
	uint32_t bfliesPerThread = 1; ///> Number of butterflies processed by a single thread
	virtual void InitializeRoots();
	virtual void EvaluatePerformance() override;
};

class NttFactor : public Ntt {
public:
	NttFactor(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride); // constructor
	virtual bool Initialize() override;
	virtual void Finalize() override;
protected:
	bool initializeTwiddleFactors = true;
	uint64_t* twiddleFactors; ///> The roots of unity (full-size buffer) used by the factor step of the assembly 4-step algorithmy
	uint64_t* invTwiddleFactors; ///> The (inverse) roots of unity (full-size buffer) used by the factor step of the assembly 4-step algorithmy
};

class Transpose : public Kernel {
public:
	Transpose(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride); // constructor
	virtual bool Initialize() override;
	virtual void Finalize() override;
protected:
	uint32_t tileSize;
	uint32_t elementsPerThread;
	bool initializeTwiddleFactors = true;
	uint64_t* twiddleFactors; ///> The roots of unity (full-size buffer) used by the factor step of the assembly 4-step algorithmy
	uint64_t* invTwiddleFactors; ///> The (inverse) roots of unity (full-size buffer) used by the factor step of the assembly 4-step algorithmy
	virtual void EvaluatePerformance() override;
};
