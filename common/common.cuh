/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <cstdarg>
#include <vector>
#include <map>

/// Macro used to check Cuda errors. Every call to Cuda function should check if any error has occured
/// Please review http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
/// I have chosen the code written by R.Crovella (http://stackoverflow.com/users/1695960/robert-crovella)
// #ifdef _DEBUG
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            throw std::runtime_error(std::string(msg) + " (" + std::string(cudaGetErrorString(__err)) + " at " + std::string(__FILE__) + " line " + std::to_string(__LINE__) + " )\n"); \
        } \
    } while (0)
// #else
// #define cudaCheckErrors(msg)
// #endif

#pragma once
enum class VerbosityLevel {
    Verbose,
    Normal,
    Silent
};

class Logger {
public:
    static Logger& getInstance();

    void setVerbosity(VerbosityLevel level);
    VerbosityLevel getVerbosity() const;

    void WriteLine(VerbosityLevel level, const std::string format, ...);
    void WriteLine(const std::string format, ...);
    void Write(VerbosityLevel level, const std::string format, ...);
    void Write(const std::string format, ...);
    void IncreaseIndent(size_t value = 4);
    void DecreaseIndent(size_t value = 4);
    void StorePerformance(const std::string& name, float performance);
    void WritePerformances(VerbosityLevel level);
    void DumpPerformances(FILE* fp);
    void ClearPerformances();
private:
    Logger();  // Private constructor (Singleton)
    VerbosityLevel verbosity = VerbosityLevel::Normal;
    void Write(VerbosityLevel level, std::string format, va_list args);
    std::map<std::string, float> performances;
    size_t indent = 0;
};

namespace common {
    extern int PerformanceLoops; // On fast GPUs you can increase this value in the INI file to get more accurate results when selecting candidates during auto-tune phase (which will be longer then)
    extern cudaStream_t stream;
    void InitThreadsAndBlocks(uint32_t size, int max, int& b, int& t);
    std::string format(const std::string format, ...); // compatible with C++11 (can be replaced by std::format with C++20)
}


