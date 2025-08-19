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
enum class TranspositionOperation {
    MUL_BEFORE,
    MUL_AFTER,
    NO_MUL
};
// Perform an out-of-place matrix transposition of a rectangular matrix n1 x n2 by loading square tiles into shared memory. Each thread is processing elementsPerThread elements.
template<int tileSize, uint32_t elementsPerThread, TranspositionOperation operation, typename T> __global__ void transpose_outofplace(const T* __restrict__ x_in, T* x_out, const T* __restrict__ factors) {
    __shared__ T tile[tileSize + 1][tileSize + 1];

    constexpr int blockHeight = tileSize / elementsPerThread;

    // Position of the tile in the global matrix
    int baseX = blockIdx.x * tileSize;
    int baseY = blockIdx.y * tileSize;
    int offset = blockIdx.z * gridDim.x * gridDim.y * tileSize * tileSize;

    // Global input coordinates
#pragma unroll
    for (int i = 0; i < elementsPerThread; ++i) {
        int x = baseX + threadIdx.x;
        int y = baseY + threadIdx.y + i * blockHeight;

        int tid = offset + y * (gridDim.x * tileSize) + x;

        if constexpr (operation == TranspositionOperation::MUL_BEFORE)
            tile[threadIdx.y + i * blockHeight][threadIdx.x] = mul_Mod(x_in[tid], factors[tid]);
        else
            tile[threadIdx.y + i * blockHeight][threadIdx.x] = x_in[tid];
    }

    __syncthreads();

    // Transpose tile coordinates
    int transBaseX = blockIdx.y * tileSize;
    int transBaseY = blockIdx.x * tileSize;

#pragma unroll
    for (int i = 0; i < elementsPerThread; ++i) {
        int x = transBaseX + threadIdx.x;
        int y = transBaseY + threadIdx.y + i * blockHeight;

        int tid = offset + y * (gridDim.y * tileSize) + x;

        if constexpr (operation == TranspositionOperation::MUL_AFTER)
            x_out[tid] = mul_Mod(tile[threadIdx.x][threadIdx.y + i * blockHeight], factors[tid]); 
        else
            x_out[tid] = tile[threadIdx.x][threadIdx.y + i * blockHeight];
    }

}

