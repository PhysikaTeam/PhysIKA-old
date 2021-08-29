#pragma once
#include"smoothFEM.h"
#include"materialold.h"
#include"sectionold.h"
#include"part.h"
#include<vector>
#include"nodemanager.h"
#include"materialmanager.h"
#include"sectionmanager.h"

using std::vector;

struct SmoothFEM;

/*
 * 光滑有限元目前不支持多gpu计算
 */
typedef struct SmoothFemManager
{
	vector<SmoothFEM> smoothFemArrayCpu_;
	SmoothFEM* smoothFemArrayGpu_;

	int smoothFemNumber_;

	Material *materialArrayGpu_;
	Section *sectionArrayGpu_;

public:
	SmoothFemManager(): smoothFemArrayGpu_(nullptr), smoothFemNumber_(0), materialArrayGpu_(nullptr),
	                    sectionArrayGpu_(nullptr)
	{
		;
	}

	void constructSmoothDomain(vector<Part> partArray_);

	void computeShapePartialDerivative(vector<Part> partArray_);

	void allocateSmoothFemArrayGpu();

	void copySmoothFromCpuToGpu(NodeManager *nodeManager);

	void linkMatAndSec(MaterialManager *matManager,SectionManager *secManager);

	void splitForMultiGpu(int totNodeNum, int num_gpu, vector<int> &elm_num_gpu);

	void freeMemory();

	~SmoothFemManager();
private:
	void constructNSFEMDomain(Part *part);

	void constructESFEMDomain(Part *part);

	void constructCSFEMDomain(Part *part);

	void constructSNSFEMDomain(Part *part);

	void computeNSFEMPartialDerivative(Part *part);

	void computeCSFEMPartialDerivative(Part *part);

	void computeESFEMPartialDerivative(Part *part);

	void computeSNSFEMPartialDerivative(Part *part);
}SmoothFemManager;