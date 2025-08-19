/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "inttypes.h"
#include <string>
#include "common/common.cuh"
#include "common/pre_calculation.cuh"
#include "common/memory_tracker.cuh"
#include "data_contexts.cuh"
#include "ntt_lengths.cuh"
#include "kernel_pool.cuh"
#include "lucas_prp.cuh"
#include "common/arithmetics.cuh"

LucasPRP::LucasPRP(MainProcessData* parentData, std::string n) {
	this->parentData = parentData;
	data = new LucasPRPData;
	data->preferredKernels = parentData->preferredKernels;
	data->PRP = parentData->PRP;
	NttLengths nttLengths;
	nttLengths.StringToLengths(n, data->lengths);
	Name = "LucasPRP " + nttLengths.LengthsToString(data->lengths);
	data->totalLength = new Length;
	nttLengths.GetTotalLength(data->lengths, data->totalLength);

	std::vector<Process*> cs;
	if (data->lengths.size() == 1) {
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_GMEM") {
			cs.push_back(new KernelPool(data, Pool::FFT_GMEM)); // FFT in global memory
		}
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_1SMEM") {
			cs.push_back(new KernelPool(data, Pool::FFT_1SMEM)); // Small FFTs that fit into shared memory
		}
	}
	if (data->lengths.size() == 2) {
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_2SMEM") {
			cs.push_back(new KernelPool(data, Pool::FFT_2SMEM)); // Medium-size FFTs with 2 loads in shared memory using simplified 4-step Bailey algorithm
		}
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_2SMEM_TRANSPOSE") {
			cs.push_back(new KernelPool(data, Pool::FFT_2SMEM_TRANSPOSE)); // Medium-size FFTs with 2 loads in shared memory using 4-step Bailey algorithm with matrix transposition
		}
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_2SMEM_STORE_TRANSPOSED") {
			cs.push_back(new KernelPool(data, Pool::FFT_2SMEM_STORE_TRANSPOSED)); // Medium-size FFTs with 2 loads in shared memory using 4-step Bailey algorithm with matrix transposition, avoiding 2 transposition steps by storing the input number as transposed
		}
	}
	if (data->lengths.size() == 3) {
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_3SMEM") {
			cs.push_back(new KernelPool(data, Pool::FFT_3SMEM)); // Large FFTs with 3 loads in shared memory using simplified 4-step Bailey algorithm
		}
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_3SMEM_TRANSPOSE") {
			cs.push_back(new KernelPool(data, Pool::FFT_3SMEM_TRANSPOSE)); // Large FFTs with 3 loads in shared memory using simplified 4-step Bailey algorithm with matrix transposition
		}
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_3SMEM_STORE_TRANSPOSED") {
			cs.push_back(new KernelPool(data, Pool::FFT_3SMEM_STORE_TRANSPOSED)); // Large FFTs with 3 loads in shared memory using simplified 4-step Bailey algorithm with matrix transposition, avoiding 2 transposition steps by storing the input number as transposed
		}
	}
	if (data->lengths.size() == 4) {
		if (parentData->preferredPool == "" || parentData->preferredPool == "FFT_4SMEM") {
			cs.push_back(new KernelPool(data, Pool::FFT_4SMEM)); // Large FFTs with 4 loads in shared memory using simplified 4-step Bailey algorithm (suboptimal)
		}
	}
	subProcessesWithCandidates.push_back(cs); // LucasPRP has only 1 sub process (KernelPool), with several candidates
}

LucasPRP::~LucasPRP() {
	for (auto length : data->lengths)
		delete length;
	data->lengths.clear();
	delete data->totalLength;
	delete data;
}

bool LucasPRP::Initialize() {
	// Allocate memory buffers on the host
	host_x = (uint64_t*)calloc(data->totalLength->n, sizeof(uint64_t));
	if (!host_x) {
		Logger::getInstance().WriteLine(VerbosityLevel::Normal, "Cannot instanciate host_x!");
		return false;
	}

	// Then prepare GPU by allocating memory buffers on device
	cudaMallocTracked((void**)&data->x, data->totalLength->n * 2 * sizeof(uint64_t)); // double size buffer for matrix transpositions
	cudaCheckErrors("cudaMalloc (x) failed!");

	cudaMallocTracked((void**)&data->weights, data->totalLength->n * sizeof(uint64_t));
	cudaCheckErrors("cudaMalloc (weights) failed!");

	cudaMallocTracked((void**)&data->unweights, data->totalLength->n * sizeof(uint64_t));
	cudaCheckErrors("cudaMalloc (unweights) failed!");

	cudaMallocTracked((void**)&data->widths, data->totalLength->n * sizeof(uint8_t));
	cudaCheckErrors("cudaMalloc (widths) failed!");

	// pre-calculation step: calculate weights and widths of the IBDWT
	int blocks, threads;
	common::InitThreadsAndBlocks(data->totalLength->n, 256, blocks, threads);
	calc_weights_widths <<< blocks, threads, 0, common::stream >>> (data->totalLength->n, parentData->exponent, data->weights, data->unweights, data->widths);
	cudaCheckErrors("Kernel launch failed: calc_weights_widths");

	InitializeTest();
	return true;
}

void LucasPRP::Finalize() {
	cudaFreeTracked(data->widths);
	cudaCheckErrors("cudaFree (widths) failed!");
	cudaFreeTracked(data->unweights);
	cudaCheckErrors("cudaFree (unweights) failed!");
	cudaFreeTracked(data->weights);
	cudaCheckErrors("cudaFree (weights) failed!");
	cudaFreeTracked(data->x);
	cudaCheckErrors("cudaFree (x) failed!");
	
	free(host_x);
	Process::Finalize();
}

void LucasPRP::InitializeTest() {
	memset(host_x, 0, data->totalLength->n * sizeof(uint64_t));
	host_x[0] = parentData->PRP ? 3 : 4; // Initial value of the PRP or Lucas-Lehmer test
	cudaMemset(data->x, 0, data->totalLength->n * 2 * sizeof(uint64_t));
	cudaCheckErrors("cudaMemset");
	cudaMemcpy(data->x, host_x, sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy (host_x) failed!");
}


void LucasPRP::Run() {
	// Reload previous run if interrupted by user (CTRL-C) or by program crash after 1 hour (residue is dumped every hour for safety reason)
	if (parentData->wasInterrupted) {
		if (LoadResidue())
			Logger::getInstance().WriteLine("Previous residue successfully loaded!");
		else {
			Logger::getInstance().WriteLine("Failed to restore previous residue, restarting from iteration 1!");
			InitializeTest();
			parentData->fromIteration = 1;
		}
	}
	else
		InitializeTest();

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent);

	MainLoop();

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&totalTime, startEvent, stopEvent);
	cudaEventDestroy(stopEvent);
	cudaEventDestroy(startEvent);

	// copy back the memory from the device to the host so we can print & test the residue
	cudaMemcpy(host_x, data->x, data->totalLength->n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy (back x) failed!");

	uint32_t maxIterations = parentData->PRP ? parentData->exponent : parentData->exponent - 2;
	// if full test requested, check the residue
	if (parentData->currentIteration > maxIterations) {
		isPrime = host_x[0] == (parentData->PRP ? 9 : 0);
		for (uint32_t i = 1; i < data->totalLength->n; i++) {
			if (host_x[i] != 0) {
				isPrime = false;
				break;
			}
		}
		if (isPrime)
			Logger::getInstance().WriteLine("\n\nM(%d) is %sprime!\n", parentData->exponent, parentData->PRP ? "probably " : "");
		else
		{
			isComposite = true;
			Logger::getInstance().WriteLine("\n\nM(%d) is composite!\n", parentData->exponent);
		}
	}
	else
		Logger::getInstance().WriteLine("\n\n%d loops performed with M(%d)\n", parentData->currentIteration - parentData->fromIteration, parentData->exponent);

	// Write the final residue to disk, environment and statistics
	if (parentData->dumpResidue || parentData->interrupted || parentData->currentIteration <= parentData->exponent - 2)
		DumpResidue();
	if (parentData->dumpEnvironment)
		DumpEnvironment();
	if (parentData->dumpStats)
		DumpStats();
}

void LucasPRP::MainLoop() {
	uint64_t j = 0;
	uint64_t k = 0;
	uint64_t pitStop =  (uint64_t)((60.0 * 1000.0 * (double)common::PerformanceLoops) / (double)Performance); // display progress every minute
	uint64_t dumpIteration = (uint64_t)((3600.0 * 1000.0 * (double)common::PerformanceLoops) / (double)Performance); // dump residue every hour

	for (parentData->currentIteration = parentData->fromIteration; parentData->currentIteration <= parentData->iterations; parentData->currentIteration++) {
		if (parentData->interrupted) break;
		j++;
		if (j == dumpIteration) {
			j = 0;
			Logger::getInstance().WriteLine("\nNow dumping intermediate residue for safety reasons...\n");
			cudaMemcpy(host_x, data->x, data->totalLength->n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
			cudaCheckErrors("cudaMemcpy (back x) failed!");
			DumpResidue();
			DumpStats();
		}

		OneIteration();

		k++;
		if (k == pitStop && Logger::getInstance().getVerbosity() != VerbosityLevel::Silent) {
			k = 0;
			DisplayPitStop();
		}
	}
}

void LucasPRP::OneIteration() {
	Process::Run();
}

void LucasPRP::DisplayPitStop() {
	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&totalTime, startEvent, stopEvent);
	double totalTime_d = static_cast<double>(totalTime);
	long long h = static_cast<long long>(totalTime_d / 3600000.0);
	long long m = static_cast<long long>((totalTime_d / 1000.0 - h * 3600) / 60);
	long long s = static_cast<long long>(totalTime_d / 1000.0) - h * 3600 - m * 60;
	double iterationsDone = static_cast<double>(parentData->currentIteration) - static_cast<double>(parentData->fromIteration) + 1.0;
	double perIter = totalTime / iterationsDone;
	double iterationsRemaining = static_cast<double>(parentData->iterations) - static_cast<double>(parentData->currentIteration);
	double remaining = perIter * iterationsRemaining;
	long long h2 = static_cast<long long>(remaining / 3600000.0);
	long long m2 = static_cast<long long>((remaining / 1000.0 - h2 * 3600) / 60);
	long long s2 = static_cast<long long>(remaining / 1000.0) - h2 * 3600 - m2 * 60;
	double pct = (100.0 * static_cast<double>(parentData->currentIteration)) / static_cast<double>(parentData->iterations);
	Logger::getInstance().WriteLine("%d%% - Elapsed %02lld:%02lld:%02lld - Remaining %02lld:%02lld:%02lld - Per iteration %.4f ms",
		static_cast<int>(pct),
		h, m, s,
		h2, m2, s2,
		perIter);
}

void LucasPRP::DumpResidue() {
	FILE* fp = fopen(parentData->dumpFilename.c_str(), "w+");
	bool transposed = (subProcesses.size() > 0) && (subProcesses[0]->Name.find("STORE_TRANSPOSE") != std::string::npos);
	uint32_t n1 = data->totalLength->n / data->lengths[0]->n;
	for (uint32_t i = 0; i < data->totalLength->n; i++) {
		uint32_t idx = transposed ? (i % n1) * data->lengths[0]->n + (i / n1) : i;
		fprintf(fp, "0x%016" PRIX64 "\n", host_x[idx]);
	}
	fclose(fp);
}

void LucasPRP::DumpEnvironment() {
	const int kb = 1024;
	const int mb = kb * kb;
	FILE* fp = fopen(parentData->envFilename.c_str(), "w+");
	fprintf(fp, "CUDA Version = %d\n", CUDART_VERSION);
	int devCount;
	cudaGetDeviceCount(&devCount);
	fprintf(fp, "CUDA devices = %d\n", devCount);
	if (devCount > 0) {
		for (int i = 0; i < devCount; ++i) {
			cudaDeviceProp props;
			cudaGetDeviceProperties(&props, i);
			fprintf(fp, "Device %d = %s\n", i, props.name);
			fprintf(fp, "  Revision = %d.%d\n", props.major, props.minor);
			fprintf(fp, "  Multiprocessors = %d\n", props.multiProcessorCount);
			fprintf(fp, "  Global memory (mb) = %d\n", (int)(props.totalGlobalMem / mb));
			fprintf(fp, "  Shared memory (kb) = %d\n", (int)(props.sharedMemPerBlock / kb));
			fprintf(fp, "  Constant memory (kb) = %d\n", (int)(props.totalConstMem / kb));
			fprintf(fp, "  Block registers = %d\n", props.regsPerBlock);
			fprintf(fp, "  Warp size = %d\n", props.warpSize);
			fprintf(fp, "  Threads per block = %d\n", props.maxThreadsPerBlock);
			fprintf(fp, "  Max block dimensions = [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
			fprintf(fp, "  Max grid dimensions = [%d, %d, %d]\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
		}
	}
	fclose(fp);
}

void LucasPRP::DumpStats() {
	FILE* fp = fopen(parentData->statFilename.c_str(), "w+");
	fprintf(fp, "Exponent = %d\n", parentData->exponent);
	if (isPrime)
		fprintf(fp, "Conclusion = %sPrime\n", parentData->PRP ? "Probably " : "");
	if (isComposite)
		fprintf(fp, "Conclusion = Composite\n");
	if (!isPrime && !isComposite)
		fprintf(fp, "Conclusion = Partial\n");
	fprintf(fp, "PRP = %d\n", parentData->PRP);
	fprintf(fp, "Iterations = %d\n", parentData->iterations);
	NttLengths nttLengths;
	fprintf(fp, "fft = %s\n", nttLengths.LengthsToString(data->lengths).c_str());
	fprintf(fp, "n = %d\n", data->totalLength->n);
	fprintf(fp, "log2_n = %d\n", data->totalLength->log2_n);
	fprintf(fp, "log3_n = %d\n", data->totalLength->log3_n);
	fprintf(fp, "log5_n = %d\n", data->totalLength->log5_n);
	Logger::getInstance().DumpPerformances(fp);
	fprintf(fp, "Pool = %s\n", subProcesses[0]->Name.c_str());
	for (int i = 0; i < subProcesses[0]->subProcesses.size(); i++)
		fprintf(fp, "Kernel%d = %s\n", i, subProcesses[0]->subProcesses[i]->Name.c_str());
	fprintf(fp, "Interrupted = %d\n", parentData->interrupted || parentData->currentIteration < parentData->iterations ? 1 : 0);
	fprintf(fp, "currentIteration = %d\n", parentData->currentIteration);
#ifdef _DEBUG
	fprintf(fp, "Warning! Following performances are not relevant as you are in DEBUG mode!\n");
#endif
	fprintf(fp, "Per iteration (ms) = %g\n", totalTime / (parentData->currentIteration - parentData->fromIteration + 1));
	if (parentData->wasInterrupted)
		fprintf(fp, "Warning! total time is not relevant since the full process has been interrupted by CTRL-C\n");
	fprintf(fp, "Total time (sec) = %g\n", totalTime / 1000);
	fclose(fp);
}

bool LucasPRP::LoadResidue()
{
	bool transposed = (subProcesses.size() > 0) && (subProcesses[0]->Name.find("STORE_TRANSPOSE") != std::string::npos);
	uint32_t n1 = data->totalLength->n / data->lengths[0]->n;
	char buf[19];
	FILE* fp = fopen(parentData->dumpFilename.c_str(), "r");
	if (fp) {
		uint32_t i = 0;
		while (fread(buf, 1, sizeof buf, fp) > 0) {
			uint32_t idx = transposed ? (i % n1) * data->lengths[0]->n + (i / n1) : i;
			host_x[idx] = strtol(buf, NULL, 16);
			i++;
		}
		fclose(fp);
		remove(parentData->dumpFilename.c_str());
		cudaMemcpy(data->x, host_x, data->totalLength->n * sizeof(uint64_t), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy (host_x) failed!");
		return true;
	}
	return false;
}


void LucasPRP::RunPerformance() {
	OneIteration();
}


