#pragma once
#include<string>
#include<vector>
#include"outputdatatype.h"
using std::string;
using std::vector;
struct FEMDynamic;

typedef struct BasicOutput
{
	int fileId_;

	double outTimeStep;

	vector<string> typeArray_;

	string name;

	int out_id;

	vector<DataType> dTypeArray_;

	vector<int> needMemPtrIndex_;

	string folderPath;

	BasicOutput();

	void verOutDataType();

	virtual void GetDataFromDomain(FEMDynamic *domain, const double time, const double dt) = 0;

	virtual void bulidDataPathToDomain(FEMDynamic *domain) = 0;

	virtual void outputDataForNeed(FEMDynamic *domain, const double time, const double dt) = 0;

	/*
	ls:2020-07-30 added / from gwp
	*/
	virtual void outputDataForNeed_bin(FEMDynamic *domain, const double time, const double dt) = 0;
	
	virtual void outputDataForNeed_bin_read(FEMDynamic* domain, const double time, const double dt) = 0;

	virtual ~BasicOutput() { ; }
}BasicOutput;