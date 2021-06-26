#pragma once
#include "section.h"

class BaseBeamCrossSection:public CrossSection
{
	double outerDiameterNode1_;

	double outerDiameterNode2_;

	double innerDiameterNode1_;

	double innerDiameterNode2_;

	double crossSectionArea_;

	double shearArea_;

	double polarIntertiaMoment_;

	double yIntertiaMoment;

	double zIntertiaMoment;

public:

	virtual ~BaseBeamCrossSection(){}

	__host__ __device__ virtual void computeInternalForce() override;
};