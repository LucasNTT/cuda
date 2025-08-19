/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#pragma once
#include <cuda_runtime.h>
#include <set>
#include <cstdio>
#include <mutex>

namespace cuda_tracker {

    // Singleton struct to hold allocations and mutex
    struct TrackerState {
        std::set<void*> allocations;
        std::mutex mtx;

        static TrackerState& instance() {
            static TrackerState instance;
            return instance;
        }
    };

    inline void log_mem(const char* label, void* ptr, size_t size, const char* file, int line) {
#ifdef MEM_TRACKER
        printf("[%s] ptr = %p (%zu bytes) at %s:%d\n", label, ptr, size, file, line);
#endif
    }

    inline cudaError_t cudaMallocTrackedImpl(void** devPtr, size_t size, const char* file, int line) {
        cudaError_t err = cudaMalloc(devPtr, size);
        if (err == cudaSuccess) {
            auto& state = TrackerState::instance();
            std::lock_guard<std::mutex> lock(state.mtx);
            state.allocations.insert(*devPtr);
            log_mem("cudaMalloc", *devPtr, size, file, line);
        }
        else {
            printf("[cudaMalloc FAILED] %zu bytes at %s:%d: %s\n", size, file, line, cudaGetErrorString(err));
        }
        return err;
    }

    inline cudaError_t cudaFreeTrackedImpl(void* devPtr, const char* file, int line) {
        cudaError_t err = cudaFree(devPtr);
        auto& state = TrackerState::instance();
        std::lock_guard<std::mutex> lock(state.mtx);
        if (err == cudaSuccess) {
            if (state.allocations.erase(devPtr)) {
                log_mem("cudaFree", devPtr, 0, file, line);
            }
            else {
                printf("[cudaFree WARNING] Unknown or double free: %p at %s:%d\n", devPtr, file, line);
            }
        }
        else {
            printf("[cudaFree FAILED] %p at %s:%d: %s\n", devPtr, file, line, cudaGetErrorString(err));
        }
        return err;
    }

    inline void reportLeaks() {
        auto& state = TrackerState::instance();
        std::lock_guard<std::mutex> lock(state.mtx);
        if (!state.allocations.empty()) {
            printf("[MEMORY LEAK DETECTED] %zu allocations not freed:\n", state.allocations.size());
            for (void* ptr : state.allocations) {
                printf("  Leaked pointer: %p\n", ptr);
            }
        }
        else {
#ifdef MEM_TRACKER
            printf("[MEMORY] All allocations freed successfully.\n");
#endif
        }
    }

} // namespace cuda_tracker

#define cudaMallocTracked(ptr, size) cuda_tracker::cudaMallocTrackedImpl((void**)(ptr), size, __FILE__, __LINE__)
#define cudaFreeTracked(ptr)         cuda_tracker::cudaFreeTrackedImpl(ptr, __FILE__, __LINE__)
#define cudaReportLeaks()           cuda_tracker::reportLeaks()

