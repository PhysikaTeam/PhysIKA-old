//#pragma once
//#include"cuda_runtime.h"
//#include"elementtypeenum.h"
//#include"node.h"
//#include "materialold.h"
//#include "sectionold.h"
//#include"hourglasscontrol.h"
//#include"gausspoint.h"
//#include"device_launch_parameters.h"
//
//struct NodeCuda;
//
//typedef struct BeamElement2nd
//{
//	int fileId_;
//	int storeId_;
//
//	int nodeIdArray_[4];
//	NodeCuda nodeArray_[4];
//
//	int gpuId_;
//
//	ElementNewType beam2ndType_;
//
//	int sectionId_;
//	int materialId_;
//
//	double dtEle_;
//
//	double lengthChart_;
//
//	double massScaleRate_;
//
//	double volume_;
//
//	double translateMass_;
//
//	double rotateMass_[3];
//
//	int gaussPointNum_;
//
//	GaussPoint *gaussPoint_;
//
//}BeamElement2nd;
//
//typedef struct SolidElement2nd
//{
//	int fileId_;
//	int storeId_;
//
//	int nodeIdArray_[20];
//	NodeCuda nodeArray_[20];
//
//	int gpuId_;
//
//	ElementNewType solid2ndType_;
//
//	int sectionId_;
//	int materialId_;
//
//	double dtEle_;
//
//	double lengthChart_;
//
//	double massScaleRate_;
//
//	double volume_;
//
//	double translateMass_;
//
//	double rotateMass_[3];
//
//	int gaussPointNum_;
//
//	GaussPoint *gaussPoint_;
//
//	__host__ __device__ void interForceTet2ndGpu(const Material* material, const Section *section, const double dt);
//
//	__host__ __device__ void calBMatrixTet2nd();
//
//	__host__ __device__ void interForceHex2ndGpu(const Material* material, const Section *section, const double dt);
//
//	__host__ __device__ void calBmaterixHex2nd();
//
//	__host__ __device__ void interForcePenta2ndGpu(const Material* material, const Section *section, const double dt);
//
//	__host__ __device__ void calBMaterixPenta2nd();
//
//	__host__ __device__ void computeSolidCharactLength2nd();
//
//	__host__ __device__ void computeSolidTimeStep(const double wave_num, const double safe_factor);
//
//	__host__ __device__ void solid2ndNodeMassIntia();
//
//	__device__ double solid2ndMassScaleGpu(const double minDt);
//
//	__host__ double solid2ndMassScaleCpu(const double minDt);
//
//	__device__ void builNodeIndex();
//
//	__host__ __device__ void bulidNodeMatch(NodeCuda *node_array_gpu);
//
//}SolidElement2nd;
//
//typedef struct ShellElement2nd
//{
//	int fileId_;
//	int storeId_;
//
//	int nodeIdArray_[8];
//	NodeCuda nodeArray_[8];
//
//	ElementNewType shell2ndType_;
//
//	int gpuId_;
//
//	int sectionId_;
//	int materialId_;
//
//	double dtEle_;
//
//	double lengthChart_;
//
//	double massScaleRate_;
//
//	double volume_;
//
//	double translateMass_;
//
//	double rotateMass_[3];
//
//	int gaussPointNum_;
//
//	GaussPoint *gaussPoint_;
//}ShellElement2nd;