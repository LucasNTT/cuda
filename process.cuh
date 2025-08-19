/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <vector>
#include <string>
#include "data_contexts.cuh"

#pragma once
class Process {
public:
	std::string Name;
	std::vector<std::vector<Process*>> subProcessesWithCandidates; ///> A list of subprocesses, each of them may have from one to many candidates
	std::vector<Process*> subProcesses; ///> The final list of subprocesses, where the best performers has been selected among the candidates
	virtual ~Process(); // destructor
	virtual bool Initialize();
	virtual void Finalize();
	virtual void Run();
	void FinalizeAll();
protected:
	const float MaxPerf = 9999999;
	float Performance = MaxPerf;
	virtual void EvaluatePerformance();
	bool GetBestPerformers(size_t indent);
	virtual void RunPerformance();
	void PrintCandidatesNames();
	void PrintSubProcessesNames();
	void DumpDevice(uint32_t n, uint64_t* d_x, std::string format, ...);
};
