/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <sstream>
#include <iomanip>
#include "common.cuh"
#include "debug.cuh"

#ifdef _DEBUG
Debug& Debug::getInstance() {
    static Debug instance;
    return instance;
}

Debug::Debug() = default;

void Debug::DumpDevice(bool after) {
	if (!active) return;
	uint64_t max = 0;
	uint64_t* h_x = (uint64_t*)calloc(n, sizeof(uint64_t));
	cudaMemcpy(h_x, x, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy (back x for dump) failed!");
	char f[100];
	sprintf(f, common::format("debug %d %d.dat", currentIteration, after).c_str());
	FILE* fp = fopen(f, "w+");
	for (uint32_t i = 0; i < n; i++) {
		fprintf(fp, "0x%016llX\n", h_x[i]);
		if (h_x[i] > max)
			max = h_x[i];
	}
	fprintf(fp, "\n\n Max = 0x%016llX\n", max);
	fclose(fp);
	free(h_x);
}

#endif // _DEBUG

