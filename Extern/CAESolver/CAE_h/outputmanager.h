#pragma once
#include <vector>
#include"string"
#include"basicoutput.h"

using std::vector;
using std::string;

struct BasicOutput;
struct FEMDynamic;

typedef struct OutputManager
{
	vector<BasicOutput*> outPutArray_;

	bool isHaveTecplot;

	bool isHaveParaView;

	double commandRFT_;

	OutputManager();

	void outputNeedDataAt(FEMDynamic *domain, const double time, const double dt);

	void getNeedDataAt(FEMDynamic *domain, const double time, const double dt);

	void buildDataPath(FEMDynamic *domain);

	void initialOutput(FEMDynamic *domain);

	~OutputManager();
}OutputManager;