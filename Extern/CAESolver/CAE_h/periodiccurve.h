#pragma once
#include"abaquscurve.h"

struct AbaqusCurve;

typedef struct PeriodicCurve:AbaqusCurve
{
	int node_num;

	double A0;

	double x_axis[MaxPoint];
	double y_axis[MaxPoint];

	double omega;

	double t0;

	PeriodicCurve();

	virtual void setCurrentValue(const double t) override;

	virtual void returnCurrentValue(const double t, double& val) override;

	virtual double returnCurrentValue(const double t) override;

	~PeriodicCurve() { ; }
}PeriodicCurve;