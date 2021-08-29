#pragma once
#include"basicoutput.h"
#include<fstream>
#include "sectionold.h"

using std::ofstream;

struct Section;

typedef struct ParaViewOutput:BasicOutput
{
	int fileCycNum_;

	int fileCycNum_bin;

	bool isNeedDisp_;

	bool isNeedVel_;

	bool isNeedAccel_;

	bool isNeedStress_;

	bool isNeedStrain_;

	bool is_binary = false;

	string outputPath;

	double intDt_;

	ofstream foutStream_;

	FILE *fid_bin;

	int elmNum_;

	int nodeOfElmSize_;

	vector<int> elmtype_;

	ParaViewOutput();

	virtual void GetDataFromDomain(FEMDynamic* domain, const double time, const double dt) override;

	virtual void bulidDataPathToDomain(FEMDynamic* domain) override;

	virtual void outputDataForNeed(FEMDynamic* domain, const double time, const double dt) override;

	void getParaViewTypeFromSection(int &elmType,Section *sec);

	/*
	ls:2020-07-22 added
	*/
	void ParaViewOutput::getParaViewTypeFromeType(int& elmType, ElementNewType eType_);

	/*
	ls:2020-07-22 added / from gwp
	*/
	virtual void outputDataForNeed_bin(FEMDynamic* domain, const double time, const double dt);

	virtual void outputDataForNeed_bin_read(FEMDynamic* domain, const double time, const double dt);

	void SwapEndfloat(double var);

	void SwapEndint(int var);

}ParaViewOutput;