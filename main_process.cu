/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#ifdef _WIN32 
#define _CRT_SECURE_NO_DEPRECATE 
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <string>
#include <cctype>
#include "SimpleIni.h"
#include "common/common.cuh"
#include "data_contexts.cuh"
#include "ntt_lengths.cuh"
#include "main_process.cuh"
#include "lucas_prp.cuh"
#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#endif

MainProcessData* data = NULL;

#ifdef _WIN32 
BOOL CtrlHandler(DWORD signal) {
	if (signal == CTRL_C_EVENT) {
		data->interrupted = true;
		Logger::getInstance().WriteLine(VerbosityLevel::Normal, "\nProcess interrupted!");
	}
	return TRUE;
}
#else
void CtrlHandler(sig_atomic_t s) {
	data->interrupted = true;
	Logger::getInstance().WriteLine(VerbosityLevel::Normal, "\nProcess interrupted!");
}
#endif

bool Confirm(const std::string& question) {
	std::string answer;
	while (true) {
		std::cout << question << " (Y/N) : ";
		std::getline(std::cin, answer);

		if (!answer.empty()) {
			char c = std::toupper(static_cast<unsigned char>(answer[0]));
			if (c == 'Y') return true;
			if (c == 'N') return false;
		}
		std::cout << "Please enter Y or N.\n";
	}
}

MainProcess::MainProcess(int argc, char** argv) {
	Name = "MainProcess";
	data = new MainProcessData;

	// LucasNTT.ini file 
	CSimpleIniA* ini = new CSimpleIniA; // SimpleIni from https://github.com/brofield/simpleini
	ini->SetUnicode();

	if (ini->LoadFile("LucasNTT.ini") < SI_OK) {
		delete ini;
		ini = NULL;
	}
	std::string verbosity = NULL == ini ? "Normal" : ini->GetValue("", "Verbosity", "Normal");
	if (verbosity == "Verbose")
		Logger::getInstance().setVerbosity(VerbosityLevel::Verbose);
	if (verbosity == "Silent")
		Logger::getInstance().setVerbosity(VerbosityLevel::Silent);
	Logger::getInstance().WriteLine("Welcome to LucasNTT, a pure-integer Lucas-Lehmer and PRP primality test for Mersenne numbers running on GPU.");
	Logger::getInstance().WriteLine("------------------------------------------------------------------------------------------------------------");
	Logger::getInstance().WriteLine("");

	cudaSetDevice(NULL == ini ? 0 : ini->GetLongValue("", "GPU", 0));
	cudaCheckErrors("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed? If yes, check your Cuda installation.");
	common::PerformanceLoops = NULL == ini ? 500 : ini->GetLongValue("", "PerformanceLoops", 500);

	data->dumpFilename = (argc > 3) ? std::string(argv[3]) + ".dat" : NULL == ini ? "LucasNTT_results.dat" : ini->GetValue("", "Dump", "LucasNTT_results.dat");
	data->statFilename = (argc > 3) ? std::string(argv[3]) + ".txt" : NULL == ini ? "LucasNTT_results.txt" : ini->GetValue("", "Stats", "LucasNTT_results.txt");
	data->envFilename = (argc > 3) ? std::string(argv[3]) + ".env" : NULL == ini ? "LucasNTT_results.env" : ini->GetValue("", "Env", "LucasNTT_results.env");
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Filename for final results: " + data->statFilename);
	// Choose to save results and other data in files
	data->dumpResidue = NULL == ini ? false : ini->GetLongValue("", "dumpResidue", 0) == 1;
	data->dumpStats = NULL == ini ? true : ini->GetLongValue("", "dumpStats", 1) == 1;
	data->dumpEnvironment = NULL == ini ? false : ini->GetLongValue("", "dumpEnvironment", 0) == 1;

	CSimpleIniA* last = new CSimpleIniA;
	last->SetUnicode();
	data->wasInterrupted = last->LoadFile(data->statFilename.c_str()) == SI_OK && last->GetLongValue("", "Interrupted", 0) == 1;
	if (data->wasInterrupted) {
		if (NULL != ini)
			delete ini;
		ini = last; // use last LucasNTT_results.txt instead of standard LucasNTT.ini
		Logger::getInstance().WriteLine("Last execution was interrupted, trying to restore last execution status...");
	}

	// Get Mersenne number to test
	data->PRP = NULL == ini ? false : ini->GetLongValue("", "PRP", 0) == 1;
	data->exponent = (argc > 1) ? std::stoi(argv[1]) : NULL == ini ? 9689 : ini->GetLongValue("", "Exponent", 9689);
	Logger::getInstance().WriteLine("Exponent = %d", data->exponent);

	// iterations to run
	uint32_t maxIterations = (data->PRP ? data->exponent : data->exponent - 2);
	data->iterations = (argc > 2) ? std::stoi(argv[2]) : NULL == ini ? maxIterations : ini->GetLongValue("", "Iterations", maxIterations);
	if (data->iterations > maxIterations || data->iterations <= 0)
		data->iterations = maxIterations; // Fix incorrect value provided by the user
	data->fromIteration = data->wasInterrupted && NULL != ini ? ini->GetLongValue("", "currentIteration", 1) : 1;
	// Get preferred processes
	data->preferredPool = NULL == ini ? "" : ini->GetValue("", "Pool", "");
	data->preferredKernels = NULL == ini ? "" : ini->GetValue("", "Kernels", "");
	data->preferredKernels = ";" + data->preferredKernels + ";";

	// Preferred fft lengths
	NttLengths nttLengths;
	std::string providedLength = (argc > 4) ? std::string(argv[4]) : NULL == ini ? "" : ini->GetValue("", "fft", "");
	if ("" == providedLength) {
		std::vector<std::string> allPossibleLengths;
		nttLengths.GetCandidates(data->exponent, allPossibleLengths);
		std::vector<Process*> cs;
		for (auto s : allPossibleLengths) {
			cs.push_back(new LucasPRP(data, s)); // create as many candidates as there are possible NTT lengths
		}
		subProcessesWithCandidates.push_back(cs); // MainProcess has only 1 sub process: LucasPRP
	}
	else {
		int diff = nttLengths.CheckProvidedLength(data->exponent, providedLength);
		if (diff != 0) {
			if (!Confirm(diff > 0 ? "\n\nThe provided length is too small. Due to the risk of overflow, the result will not be reliable.\nDo you still want to process?" : "\n\nThe provided length is not optimal. The process can be accelerated.\nDo you still want to process?"))
				throw std::runtime_error("Process cancelled by user. Invalid FFT length.");
		}
 		std::vector<Process*> cs;
		cs.push_back(new LucasPRP(data, providedLength)); // If the NTT length is specified, there is only one candidate 
		subProcessesWithCandidates.push_back(cs); // MainProcess has only 1 sub process: LucasPRP
	}
	Logger::getInstance().WriteLine("");
	if (NULL != ini)
		delete ini;
}

MainProcess::~MainProcess() {
	cudaDeviceReset();
	delete data;
}

void MainProcess::Run() {
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "Here are the candidates to run this test:");
	PrintCandidatesNames();
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "");

	cudaStreamCreate(&common::stream);
	cudaCheckErrors("cudaStreamCreate");

	GetBestPerformers(0);
	Logger::getInstance().WriteLine(VerbosityLevel::Normal, "\nSummary of the performances of the different processes on %d loops:", common::PerformanceLoops);
	Logger::getInstance().WritePerformances(VerbosityLevel::Normal);
	// Check that a valid branch exists
	if (subProcesses.size() == 1 && subProcesses[0]->subProcesses.size() == 1 && subProcesses[0]->subProcesses[0]->subProcesses.size() > 0) {

		Logger::getInstance().WriteLine("\nThis test will run using the following processes:");
		Logger::getInstance().WriteLine(VerbosityLevel::Normal, Name);
		PrintSubProcessesNames();

		Logger::getInstance().WriteLine("\n\n----------------------------------------------");
		Logger::getInstance().WriteLine("Now testing M(q) = 2^q - 1 where q = %d", data->exponent);
		if (data->PRP)
			Logger::getInstance().WriteLine("With PRP pseudo-primality test");
		else
			Logger::getInstance().WriteLine("With Lucas-Lehmer primality test");
		Logger::getInstance().WriteLine("From iteration %d to %d\n", data->fromIteration, data->iterations);

#ifdef _WIN32 
		SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);
#else
		signal(SIGINT, CtrlHandler);
#endif

		Process::Run();
	}

	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "\nFinalization...");
	FinalizeAll();
	cudaStreamDestroy(common::stream);
	cudaCheckErrors("cudaStreamDestroy");
	Logger::getInstance().WriteLine(VerbosityLevel::Verbose, "End.");
}

