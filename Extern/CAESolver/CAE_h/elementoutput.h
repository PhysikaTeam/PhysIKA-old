#pragma once
#include"basicoutput.h"
#include"set.h"

struct Set;

typedef struct ElementOutput :BasicOutput
{
	Set* elementSet_;

	double* needMemoryCpu_;
	double* needMemoryGpu_;

	int needMemSize_;

	ElementOutput();

	virtual void GetDataFromDomain(FEMDynamic* domain, const double time, const double dt) override;

	virtual void bulidDataPathToDomain(FEMDynamic* domain) override;

	virtual void outputDataForNeed(FEMDynamic* domain, const double time, const double dt) override;

	/*
	ls:2020-07-22 added / from gwp
	*/
	virtual void outputDataForNeed_bin(FEMDynamic* domain, const double time, const double dt);

	virtual void outputDataForNeed_bin_read(FEMDynamic* domain, const double time, const double dt);


	virtual ~ElementOutput();
}ElementOutput;