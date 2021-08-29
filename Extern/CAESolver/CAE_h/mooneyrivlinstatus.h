#pragma once
#include"materialnew.h"

typedef struct MooneyRivlinStatus:MaterialStatus
{
	double deformGrandient_[3][3];
}MooneyRivlinStatus;