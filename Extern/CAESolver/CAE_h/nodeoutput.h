#pragma once
#include"basicoutput.h"
#include"set.h"
#include "rigid.h"

struct Set;
struct RigidCuda;

typedef struct NodeOutput :BasicOutput
{
	Set* nodeSet_;
	RigidCuda* rigid;

	double *needMemoryGpu;
	double *needMemoryCpu;

	vector<NodeCuda**> nodeInMultiGpu;
	vector<int> nodeNumInMultiGpu;

	int needMemSize_;

	NodeOutput();

	void GetDataFromDomain(FEMDynamic *domain, const double time, const double dt)override;

	void bulidDataPathToDomain(FEMDynamic *domain)override;

	virtual void outputDataForNeed(FEMDynamic *domain, const double time, const double dt) override;

	/*
	ls:2020-07-22 added / from gwp
	*/
	virtual void outputDataForNeed_bin(FEMDynamic* domain, const double time, const double dt);

	virtual void outputDataForNeed_bin_read(FEMDynamic* domain, const double time, const double dt);

	virtual ~NodeOutput();
}NodeOutPut;