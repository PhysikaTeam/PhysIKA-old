#pragma once
#include"basicoutput.h"
#include<fstream>
#include "sectionold.h"

using std::ofstream;
using std::vector;
using std::map;

struct Section;

typedef struct PartParaViewOutput :BasicOutput
{
	int fileCycNum_part;

	bool isNeedDisp_;

	bool isNeedVel_;

	bool isNeedAccel_;

	bool isNeedStress_;

	bool isNeedStrain_;

	string outputPath;

	double intDt_;

	ofstream foutStream_;

	int elmNum_;

	int nodeOfElmSize_;
	
	vector<int> elmtype_;

	vector<int> elmNum_part;

	vector<int> nodeOfElmSize_part;
	
	vector<vector<int>> elmtype_part;

	vector<map<int, int>> fileIdToPartId;

	PartParaViewOutput();

	void GetDataFromDomain(FEMDynamic* domain, const double time, const double dt) override;

	void bulidDataPathToDomain(FEMDynamic* domain) override;

	void outputDataForNeed(FEMDynamic* domain, const double time, const double dt) override;

	void outputDataForNeed(Part *part, int i, FEMDynamic* domain, const double time, const double dt);

	void getParaViewTypeFromSection(int &elmType, Section *sec);


	virtual void outputDataForNeed_bin(FEMDynamic *domain, const double time, const double dt) {};

	virtual void outputDataForNeed_bin_read(FEMDynamic* domain, const double time, const double dt) {};

}PartParaViewOutput;