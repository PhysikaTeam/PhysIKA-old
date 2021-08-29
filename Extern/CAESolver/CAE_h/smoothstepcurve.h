#pragma once
#include"abaquscurve.h"

struct AbaqusCurve;

typedef struct SmoothStepCurve:AbaqusCurve
{
	int node_num;

	double x_axis[MaxPoint];
	double y_axis[MaxPoint];

	SmoothStepCurve();

	virtual void setCurrentValue(const double t) override;

	virtual void returnCurrentValue(const double t, double& val) override;

	virtual double returnCurrentValue(const double t) override;

	~SmoothStepCurve() { ; }
}SmoothStepCurve;