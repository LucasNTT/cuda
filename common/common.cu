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
#include <algorithm>
#include "common.cuh"

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() = default;

void Logger::setVerbosity(VerbosityLevel level) {
    verbosity = level;
}

VerbosityLevel Logger::getVerbosity() const {
    return verbosity;
}

void Logger::IncreaseIndent(size_t value) {
    indent += value;
}
void Logger::DecreaseIndent(size_t value) {
    indent -= value;
}

void Logger::WriteLine(VerbosityLevel level, const std::string format, ...) {
    va_list args;
    va_start(args, format);

    Write(level, std::string(indent, ' ') + format + "\n", args);

    va_end(args);
}

void Logger::WriteLine(const std::string format, ...) {
    va_list args;
    va_start(args, format);

    Write(VerbosityLevel::Normal, std::string(indent, ' ') + format + "\n", args);  // Default to Normal verbosity

    va_end(args);
}

void Logger::Write(VerbosityLevel level, const std::string format, ...) {
    va_list args;
    va_start(args, format);

    Write(level, format, args);

    va_end(args);
}

void Logger::Write(const std::string format, ...) {
    va_list args;
    va_start(args, format);

    Write(VerbosityLevel::Normal, format, args);  // Default to Normal verbosity

    va_end(args);
}


void Logger::Write(VerbosityLevel level, const std::string format, va_list args) {
    if (verbosity <= level)
        vprintf(format.c_str(), args);
}

void Logger::StorePerformance(const std::string& name, float performance) {
    performances[name] = performances.count(name) ? std::min(performances[name], performance) : performance;
}

void Logger::ClearPerformances() {
    performances.clear();
}

void Logger::WritePerformances(VerbosityLevel level) {
    size_t maxNameLength = 0;
    for (const auto& p : performances) {
        if (p.first.length() > maxNameLength)
            maxNameLength = p.first.length();
    }

    for (const auto& p : performances) {
        std::ostringstream oss;
        oss << std::left << std::setw(static_cast<int>(maxNameLength)) << p.first
            << " = "
            << std::right << std::fixed << std::setw(8) << std::setprecision(3)
            << p.second << "ms";
        WriteLine(level, "%s", oss.str().c_str());
    }
}

void Logger::DumpPerformances(FILE* fp) {
    for (const auto& p : performances) {
        fprintf(fp, "Candidate Performance (in ms for %d loops): %s = %g\n", common::PerformanceLoops, p.first.c_str(), p.second);
    }
}


namespace common
{
    int PerformanceLoops = 500;
    cudaStream_t stream;

    void InitThreadsAndBlocks(uint32_t size, int max, int& b, int& t) {
        t = max;
        b = size / t;
        if (b == 0) {
            b = 1;
            t = size;
        }
    }
    std::string format(const std::string format, ...) {
        va_list args;

        // compute buffer size
        va_start(args, format);
        int size = std::vsnprintf(nullptr, 0, format.c_str(), args) + 1;
        va_end(args);
        if (size <= 0) {
            throw std::runtime_error("Erreur de formatage");
        }

        // 2. format string
        std::vector<char> buffer(size);
        va_start(args, format);
        std::vsnprintf(buffer.data(), size, format.c_str(), args);
        va_end(args);

        return std::string(buffer.data(), buffer.data() + size - 1); // Supprime le '\0' final
    }
}


