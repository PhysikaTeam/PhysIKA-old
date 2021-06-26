#pragma once
#include"materialnew.h"

typedef struct ViscoElasticityMaterial:MaterialNew
{
	double shortTimeShearModulus_;
	double longTimeShearModules_;
	double decayConstant_;

	ViscoElasticityMaterial();

	ViscoElasticityMaterial(Material *oldMat);
};