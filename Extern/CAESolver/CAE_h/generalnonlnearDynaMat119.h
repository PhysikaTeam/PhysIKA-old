#pragma once
#include"materialnew.h"

typedef struct GeneralNonlnearMaterial119 :MaterialNew
{
	int loadCurveIdTranslateR_;
	int loadCurveIdTranslateS_;
	int loadCurveIdTranslateT_;

	int loadCurveIdRotateR_;
	int loadCurveIdRotateS_;
	int loadCurveIdRotateT_;

	double2 loadCurveTranslateR_[CurveMaxNodeNum];
	double2 loadCurveTranslateS_[CurveMaxNodeNum];
	double2 loadCurveTranslateT_[CurveMaxNodeNum];

	double2 loadCurveRotateR_[CurveMaxNodeNum];
	double2 loadCurveRotateS_[CurveMaxNodeNum];
	double2 loadCurveRotateT_[CurveMaxNodeNum];

	__host__ __device__ GeneralNonlnearMaterial119();

	__host__ __device__ GeneralNonlnearMaterial119(Material *oldMat);
}GeneralNonlnearMaterial119;