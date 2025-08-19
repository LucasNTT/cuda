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
#include <cstdarg>
#include <stdexcept>
#include "common/common.cuh"
#include "process.cuh"

Process::~Process() {
	for (auto subProcess : subProcessesWithCandidates) { // the list has been cleared before, when selecting the best performers
		for (auto candidate : subProcess)
			delete candidate;
		subProcess.clear();
	}
	subProcessesWithCandidates.clear();
	for (auto subProcess : subProcesses)
		delete subProcess;
	subProcesses.clear();
}

bool Process::Initialize() {
	return true;
}

void Process::Finalize() {

}

void Process::Run() {
	for (auto process : subProcesses) {
		process->Run();
	}
}

bool Process::GetBestPerformers(size_t indent) {
	// Recursive function that evaluate all branches to select the best processes
	// A single branch may fail (result = false) when the Initialize() returns false, or when there is no valid candidates further in the branch.
	// Input: subProcessesWithCandidates, a list of subprocesses containing one to many candidates
	// Output: subProcesses, a list of subprocesses
	Process* best = NULL;
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Initializing process...", Name.c_str());
	Logger::getInstance().IncreaseIndent(3 + Name.size());
	bool result = Initialize();
	Logger::getInstance().DecreaseIndent(3 + Name.size());
	if (result) {
		bool hasSeveralCandidates = false;
		for (auto SubProcessWithCandidates : subProcessesWithCandidates) {
			if (SubProcessWithCandidates.size() > 1) {
				hasSeveralCandidates = true;
				std::string msg = common::format("[%s] Now selecting the best performers among %d candidates", Name.c_str(), SubProcessWithCandidates.size());
				for (auto process : SubProcessWithCandidates) {
					msg += common::format(", %s", process->Name.c_str());
				}
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, msg);
			}
			best = NULL;
			Logger::getInstance().IncreaseIndent();
			for (auto process : SubProcessWithCandidates) { // loop over candidates
				if (process->GetBestPerformers(indent + 4)) {
					Logger::getInstance().IncreaseIndent(3 + process->Name.size());
					process->EvaluatePerformance();
					if (process->Performance < MaxPerf) 
					{
						Logger::getInstance().StorePerformance(process->Name, process->Performance);
						Logger::getInstance().DecreaseIndent(3 + process->Name.size());
						Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Performance = %gms", process->Name.c_str(), process->Performance);
						if (best == NULL) {
							Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] First process, this is the champion so far...", process->Name.c_str());
							best = process;
						}
						else {
							if (process->Performance < best->Performance) {
								Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] We have a new champion!", process->Name.c_str());
								Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Finalization.", best->Name.c_str());
								best->FinalizeAll();
								delete best;
								best = process;
							}
							else {
								Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Not a champion... Finalization.", process->Name.c_str());
								process->FinalizeAll();
								delete process;
							}
						}
					}
					else {
						Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Not a valid configuration... Finalization.", process->Name.c_str());
						process->FinalizeAll();
						delete process;
					}
					
				}
				else {
					Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Dead branch", process->Name.c_str());
					delete process;
				}
			}
			if (best != NULL) {
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[%s] Champion successfully stored.", best->Name.c_str());
				subProcesses.push_back(best);
			}
			Logger::getInstance().DecreaseIndent();
			if (best == NULL) {
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, Name + ": cannot find a suitable candidate to run with this configuration.");
				Finalize();
				result = false;
			}
			SubProcessWithCandidates.clear();
		}
		subProcessesWithCandidates.clear();
		if (result && hasSeveralCandidates) { // so far, the performance of one process is not equal to the sum of the performances if the sub-processes, because several candidates have been evaluated
			//EvaluatePerformance(); // so once the best performers have been selected in the children, we get the final performances of the parent
			//Logger::getInstance().StorePerformance(Name, Performance);
		}
	}
	return result;
}

void Process::RunPerformance() {
	Run();
}

void Process::EvaluatePerformance() {
	// Run a test on a few loop
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent);

	for (int i = 0; i < common::PerformanceLoops; i++) { // iterations to evaluate the performance of this kernel
		RunPerformance();
	}

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&Performance, startEvent, stopEvent);
	cudaEventDestroy(stopEvent);
	cudaEventDestroy(startEvent);
}

void Process::PrintCandidatesNames() {
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, Name);
	for (auto subProcess : subProcessesWithCandidates) {
		for (std::size_t i = 0, e = subProcess.size(); i != e; ++i) {
			Logger::getInstance().IncreaseIndent();
			if (subProcess.size() == 1)
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[Sub-process] ");
			else
				Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "[Candidate %d] ", i + 1);
			subProcess[i]->PrintCandidatesNames();
			Logger::getInstance().DecreaseIndent();
		}
	}
}

void Process::PrintSubProcessesNames() {
	Logger::getInstance().IncreaseIndent();

	float totalPerf = 0;
	if (subProcesses.size() > 1) {
		for (auto subProcess : subProcesses) {
			totalPerf += subProcess->Performance;
		}
	}
	for (auto subProcess : subProcesses) {
		std::string info = subProcesses.size() > 1 ? common::format("%s (%.2f%%%%)", subProcess->Name.c_str(), 100 * subProcess->Performance / totalPerf) : subProcess->Name; // double-escape % because the string is passed again in another variadic function
		Logger::getInstance().WriteLine(VerbosityLevel::Normal, info);
		subProcess->PrintSubProcessesNames();
	}
	Logger::getInstance().DecreaseIndent();
}

void Process::FinalizeAll() {
	Finalize();
	for (auto process : subProcesses) { 
		process->FinalizeAll();
	}
}

char* replace_char(char* str, char find, char replace) {
	char* current_pos = strchr(str, find);
	while (current_pos) {
		*current_pos = replace;
		current_pos = strchr(current_pos, find);
	}
	return str;
}

void Process::DumpDevice(uint32_t n, uint64_t* d_x, std::string format, ...)
{
	char filename[1024]; // large enough to avoid overflow
	va_list args;
	va_start(args, format);
	vsnprintf(filename, sizeof(filename), format.c_str(), args);
	va_end(args);

	uint64_t max = 0, min = UINT64_MAX, sum = 0, nbZeros = 0;
	uint64_t* x = (uint64_t*)calloc(n, sizeof(uint64_t));
	if (!x) return;
	cudaMemcpy(x, d_x, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy (back x for dump) failed!");
	FILE* fp = fopen(filename, "w+");
	for (uint32_t i = 0; i < n; i++) {
		fprintf(fp, "0x%016" PRIX64 "\n", x[i]);
		if (x[i] > max)
			max = x[i];
		if (x[i] < min)
			min = x[i];
		if (x[i] == 0)
			nbZeros++;
		sum += x[i];
	}
	fprintf(fp, "\nZeros = %" PRIu64 "\n", nbZeros);
	fprintf(fp, "Min = 0x%016" PRIX64 "\n", min);
	fprintf(fp, "Max = 0x%016" PRIX64 "\n", max);
	fprintf(fp, "Sum = 0x%016" PRIX64 "\n", sum);
	fclose(fp);
	free(x);
}

