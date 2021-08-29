#pragma once
#include"cuda_runtime.h"

struct GaussPoint;

struct  MaterialStatus
{
	GaussPoint* gaussPoint_;

	/*
	* 应力按以下规则存储: 0 1 2 3 4 5对应于 xx yy zz xy yz zx
	*/
	double stress_[6];

	double pressure_;

	/*
	* 应变按以下规则存储: 0 1 2 3 4 5对应于 xx yy zz xy yz zx
	*/
	double strain_[6];

public:
	__host__ __device__ MaterialStatus();

	__host__ __device__ void linkGaussPoint(GaussPoint *parentGauss);
};
