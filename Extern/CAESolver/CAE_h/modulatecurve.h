#pragma once
#include"abaquscurve.h"

struct AbaqusCurve;

typedef struct ModulateCurve:AbaqusCurve
{
	double A0;
	
	double A;

	double t0;
	
	double omega1;
	
	double omega2;

	ModulateCurve();

	void returnCurrentValue(const double t, double& val) override;

	double returnCurrentValue(const double t) override;

	void setCurrentValue(const double t) override;

	~ModulateCurve(){}

}ModulateCurve;