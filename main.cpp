///
/// LucasNTT G.HERAULT
/// Version 2.2
/// 

/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <stdio.h>
#include <stdexcept>
#include "common/common.cuh"
#include "common/memory_tracker.cuh"
#include "main_process.cuh"

int main(int argc, char** argv) {
	try {
		Logger::getInstance().setVerbosity(VerbosityLevel::Normal);
		MainProcess* mainProcess = new MainProcess(argc, argv);
		mainProcess->Run();
		delete mainProcess;
	}
	catch (const std::runtime_error& e) {
		printf("%s\n", e.what());
		return 1;
	}
	catch (...) {
		printf("Unhandled exception! Abort!\n"); 
		return 1;
	}
	cudaReportLeaks();
	return 0;

}