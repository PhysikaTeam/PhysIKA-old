#pragma once
#define MaxPoint 128

typedef struct AbaqusCurve 
{
	double scaleX_;
	double scaleY_;
	
	double shiftX_;
	double shiftY_;

	double value;

	AbaqusCurve();

	virtual void setCurrentValue(const double t)=0;

	virtual void returnCurrentValue(const double t,double &val)=0;

	virtual double returnCurrentValue(const double t)=0;

	virtual ~AbaqusCurve() { ; }
}AbaqusCurve;