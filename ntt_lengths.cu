/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "common/arithmetics.cuh"
#include "data_contexts.cuh"
#include "ntt_lengths.cuh"

std::vector<std::string> NttLengths::split(const std::string& s, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item, delim)) {
		result.push_back(item);
	}

	return result;
}

Length* NttLengths::StringToLength(std::string description) {
	uint32_t factor = 1;
	if (std::toupper(description.back()) == 'K') {
		factor = 1024;
		description.pop_back();
	}
	if (std::toupper(description.back()) == 'M') {
		factor = 1024 * 1024;
		description.pop_back();
	}
	Length* result = new Length;
	result->n = std::stoi(description) * factor;
	FillLength(result);
	return result;
}

void NttLengths::FillLength(Length* length) {
	uint32_t n = length->n;
	while (n % 2 == 0) {
		length->log2_n++;
		n /= 2;
	}
	while (n % 3 == 0) {
		length->log3_n++;
		n /= 3;
	}
	while (n % 5 == 0) {
		length->log5_n++;
		n /= 5;
	}
	if (n != 1)
		throw std::runtime_error("Invalid NTT length provided! Should a multiple of 2, 3 and 5.\n\n");
	if (length->log3_n > 1)
		throw std::runtime_error("Invalid NTT length provided! Cannot use exponent of 3 greater than 1.\n\n");
	if (length->log5_n > 1)
		throw std::runtime_error("Invalid NTT length provided! Cannot use exponent of 5 greater than 1.\n\n");
}

void NttLengths::StringToLengths(std::string description, std::vector<Length*>& lengths) {
	std::vector<std::string> list = split(description, ':');
	for (auto part : list) {
		lengths.push_back(StringToLength(part));
	}
}

int NttLengths::CheckProvidedLength(uint32_t exponent, std::string description) {
	uint32_t log_n = 0;
	std::vector<std::string> list = split(description, ':');
	for (auto part : list) {
		Length* length = StringToLength(part);
		log_n += length->log2_n;
		delete length;
	}
	return GetNttSecuredLength(exponent) - log_n;
}

uint32_t NttLengths::GetNttSecuredLength(uint32_t exponent) {
	// First we get the smallest integer n where q/n < 32 (if n is too small, q/n will be bigger than 32, and 2nm^2 will be bigger than our modulo 2^64-2^32+1)
	double log_n = ceil(log2(1.0 * exponent / 32));
	// We know now that each item m is lower than 2^32
	double m = exp2(ceil(1.0 * exponent / exp2(log_n))) - 1; // m = 2^(q/n) - 1, m < 2^32
	// Now we can check if 2nm^2 < p (if this is not the case, we were too optimistic and we need to increase log_n)
	if (m * m >= 1.0 * MODULO / (2 * exp2(log_n))) {
		log_n++;
	}
	return (uint32_t)log_n;
}

void NttLengths::GetCandidates(uint32_t exponent, std::vector<std::string>& candidates) {
	uint32_t log2_n = GetNttSecuredLength(exponent);
	candidates.push_back(std::to_string(exp2(log2_n)));

	if (log2_n >= 10 && log2_n <= 24) { // FFT-2SMEM
		uint32_t a = log2_n / 2;
		candidates.push_back(std::to_string(exp2(a)) + ":" + std::to_string(exp2(log2_n - a)));
		if (log2_n - a != a)
			candidates.push_back(std::to_string(exp2(log2_n - a)) + ":" + std::to_string(exp2(a)));
	}
	if (log2_n == 23) { // FFT-3SMEM
		candidates.push_back("256:256:128");
		candidates.push_back("512:256:64");
	}
	if (log2_n == 24) {
		candidates.push_back("512:256:128");
		candidates.push_back("512:512:64");
	}
	if (log2_n == 25) {
		candidates.push_back("512:512:128");
		candidates.push_back("1024:256:128");
		candidates.push_back("1024:512:64");
	}
	if (log2_n == 26) {
		candidates.push_back("512:512:256");
		candidates.push_back("1024:256:256");
		candidates.push_back("1024:512:128");
	}
}

void NttLengths::GetTotalLength(std::vector<Length*> lengths, Length*& length) {
	// Calculate totalLength. For example for a fft with two lengths 64:64, totalLength is 4096
	length->n = 1;
	length->log2_n = 0;
	length->log3_n = 0;
	length->log5_n = 0;
	for (Length* l : lengths) {
		length->n *= l->n;
		length->log2_n += l->log2_n;
		length->log3_n += l->log3_n;
		length->log5_n += l->log5_n;
	}
}

std::string NttLengths::LengthsToString(std::vector<Length*> lengths) {
	std::string result = "";
	for (Length* length : lengths) {
		if (result != "")
			result += ":";
		result += std::to_string(length->n);
	}
	return result;
}


