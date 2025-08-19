/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <stdint.h>

__global__ void calc_weights_widths(uint32_t n, uint32_t exponent, uint64_t* weights, uint64_t* unweights, uint8_t* widths);
__global__ void calc_weights_widths_stride(uint32_t n, uint32_t exponent, uint32_t stride, uint64_t* weights, uint64_t* unweights, uint8_t* widths);
__global__ void calc_roots(uint32_t n, uint64_t* roots, uint64_t* invRoots);
__global__ void calc_roots_pow(uint32_t n, uint64_t* roots, uint64_t* invRoots, uint32_t pow);
__global__ void calc_brev_factors(uint32_t n, uint32_t log_n, uint32_t stride, uint64_t* twiddleFactors, uint64_t* invTwiddleFactors);

