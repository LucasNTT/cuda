/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <string>
#include <vector>
#include "data_contexts.cuh"

#pragma once
class NttLengths {
	// Helper that provides useful methods around fft lengths
public:
	uint32_t GetNttSecuredLength(uint32_t exponent);
	void GetCandidates(uint32_t exponent, std::vector<std::string>& candidates);
	void StringToLengths(std::string description, std::vector<Length*>& lengths);
	std::string LengthsToString(std::vector<Length*> lengths);
	void GetTotalLength(std::vector<Length*> lengths, Length*& length);
	int CheckProvidedLength(uint32_t exponent, std::string description);
private:
	std::vector<std::string> split(const std::string& s, char delim);
	Length* StringToLength(std::string description);
	void FillLength(Length* length);
};
