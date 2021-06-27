#pragma once
#include"materialstatus.h"

typedef struct SoilFoamStatus:MaterialStatus
{
	double press_;
	double volumeStrain_;
	double initialVolume_;
	bool isFailure;
}SoilFoamStatus;