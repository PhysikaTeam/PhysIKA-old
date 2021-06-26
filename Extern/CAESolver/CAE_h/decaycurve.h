#pragma once
#include"abaquscurve.h"

struct AbaqusCurve;

typedef struct DecayCurve:AbaqusCurve
{
	double A0;
	
	double A;

	double td;

	double t0;

	DecayCurve();

	double returnCurrentValue(const double t) override;

	void returnCurrentValue(const double t, double& val) override;

	void setCurrentValue(const double t) override;

	~DecayCurve() { ; }
}DecayCurve;