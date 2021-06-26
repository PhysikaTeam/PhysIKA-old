#pragma once
#include"cuda_runtime.h"
#include"materialnew.h"

struct MaterialNew;

typedef struct HourglassControl
{
	////ls:2020-03-18
	//int IHQ;
	//double QH;
	////

	int file_id;

	int hourg_control_type;  //IHQ
	double hourg_coeffi;   //QH

	HourglassControl();

	HourglassControl(int f_id, int hgType, double hgCoe);

	virtual ~HourglassControl() { ; }

	__host__ __device__ void computeHourglassValueForSolid(const MaterialNew *mat,double *hgValue,double bibi,double volume,double dt)const;
}HourglassControl;