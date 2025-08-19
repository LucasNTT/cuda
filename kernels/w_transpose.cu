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
#include "../core/transpose_oop.cuh"
#include "w_transpose.cuh"

w_transpose::w_transpose(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride)
	: Transpose(lucasPRPData, length, count, stride) {
	Name = "w_transpose";
	initializeTwiddleFactors = false;
}

void w_transpose::Run() {
	dim3 threads(tileSize, tileSize / elementsPerThread);
	dim3 blocks(data->stride / tileSize, data->length->n / tileSize, data->count);
	switch (100 * tileSize + elementsPerThread) {
	case 401: transpose_outofplace<4, 1, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 402: transpose_outofplace<4, 2, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 404: transpose_outofplace<4, 4, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 408: transpose_outofplace<4, 8, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 416: transpose_outofplace<4, 16, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 801: transpose_outofplace<8, 1, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 802: transpose_outofplace<8, 2, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 804: transpose_outofplace<8, 4, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 808: transpose_outofplace<8, 8, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 816: transpose_outofplace<8, 16, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 1601: transpose_outofplace<16, 1, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 1602: transpose_outofplace<16, 2, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 1604: transpose_outofplace<16, 4, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 1608: transpose_outofplace<16, 8, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 1616: transpose_outofplace<16, 16, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 3201: transpose_outofplace<32, 1, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 3202: transpose_outofplace<32, 2, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 3204: transpose_outofplace<32, 4, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 3208: transpose_outofplace<32, 8, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	case 3216: transpose_outofplace<32, 16, TranspositionOperation::MUL_BEFORE, uint64_t> <<< blocks, threads, 0, common::stream >>> (lucasPRPData->x, lucasPRPData->x + lucasPRPData->totalLength->n, lucasPRPData->weights); break;
	}
	cudaCheckErrors("Kernel launch failed: weight_transpose");
}
