/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "../process.cuh"
#include "../data_contexts.cuh"
#include "../kernel.cuh"

#pragma once
class transpose_tw : public Transpose {
public:
	transpose_tw(LucasPRPData* lucasPRPData, Length* length, uint32_t count, uint32_t stride); // constructor
	virtual void Run() override;
};
