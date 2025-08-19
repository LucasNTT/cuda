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
///
/// Atomic function that performs an add and a bitwise And in the same "transaction"
/// meaning that we are sure that the value we have read has not been overwritten by another thread
/// Based on the hint given in the Cuda Programming Guide (http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)
///
/// 

inline __device__ uint64_t atomicAddAnd(uint64_t* address, uint64_t valAdd, uint64_t valAnd)
{
	unsigned long long int* addr = reinterpret_cast<unsigned long long int*>(address);
	unsigned long long int old = *addr, assumed;
	do {
		assumed = old;
		old = atomicCAS(addr, assumed, (assumed + valAdd) & valAnd);
	} while (assumed != old);
	return old;
}

