#pragma once
#include"curve.h"
#include<vector>
#include<map>
using std::map;
using std::vector;

typedef struct  CurveManager
{
	vector<Curve> curve_array;
	vector<double> curveValue;
	Curve *curve_array_gpu;
	int totCurveNum;
	map<int, Curve*> curve_map;

	CurveManager();

	/**
	按时间设定曲线值
	*/
	void set_curve_value(const double time);

	/**
	曲线复制到gpu
	*/
	void curve_cpy_gpu();

	/**
	gpu曲线清除
	*/
	void gpuCurveClear();

	/**
	将曲线与id联系起来
	*/
	void curveLinkId();

	Curve* returnCurve(const int id);

	~CurveManager();
} CurveManager;

