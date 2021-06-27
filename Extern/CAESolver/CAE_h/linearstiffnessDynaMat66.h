#pragma once
#include"materialnew.h"

typedef struct LinearStiffnessMaterial66:MaterialNew
{
	double translateStiffnessR_;
	double translateStiffnessS_;
	double translateStiffnessT_;

	double rotationalStiffnessR_;
	double rotationalStiffnessS_;
	double rotationalStiffnessT_;

	double translateDamperR_;
	double translateDamperS_;
	double translateDamperT_;

	double rotationalDamperR_;
	double rotationalDamperS_;
	double rotationalDamperT_;

	__host__ __device__  LinearStiffnessMaterial66();
	
	__host__ __device__  LinearStiffnessMaterial66(Material* oldMat);
}LinearStiffnessMaterial66;