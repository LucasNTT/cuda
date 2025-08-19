/*
Copyright (C) 2025 Guillaume P. Hérault (https://github.com/LucasNTT/LucasNTT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

#include "process.cuh"
#include "data_contexts.cuh"

#pragma once
class LucasPRP : public Process {
public:
	LucasPRP(MainProcessData* parentData, std::string n); // contructor
	~LucasPRP() override; // destructor
	virtual bool Initialize() override;
	virtual void Finalize() override;
	virtual void Run() override;
	float totalTime = 0;
	LucasPRPData* data = NULL;
protected:
	virtual void RunPerformance() override;
private:
	uint64_t* host_x;   ///> Copy of the device buffer on the host
	MainProcessData* parentData = NULL;
	cudaEvent_t startEvent, stopEvent;
	bool isPrime = false, isComposite = false;
	void MainLoop();
	void OneIteration();
	void DisplayPitStop();
	void InitializeTest();
	void DumpResidue();
	void DumpEnvironment();
	void DumpStats();
	bool LoadResidue();
};
