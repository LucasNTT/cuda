/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <iostream>
#include <string>
#include <cstdarg>
#include <vector>
#include <map>

#pragma once

#ifdef _DEBUG
class Debug {
public:
    static Debug& getInstance();

    bool active = false;
    uint32_t fromIteration = 0;
    uint32_t toIteration = 0;
    uint32_t currentIteration = 0;
    bool dumpBeforeRun = false;
    bool dumpAfterRun = false;
    uint32_t offset = 0;
    std::string debugKernels;
    uint64_t* x;
    uint32_t n;
    void DumpDevice(bool after);
private:
    Debug();  // Private constructor (Singleton)
};
#endif // _DEBUG

